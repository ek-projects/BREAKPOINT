"""Nebius Token Factory / NVIDIA Nemotron integration for BREAKPOINT.

Nemotron is deliberately *not* trusted to execute arbitrary probes. It turns
structured incident context into hypotheses and a ranked shortlist drawn from a
safe probe registry. BREAKPOINT's deterministic planner remains the authority
that chooses the next probe from the validated shortlist.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
DEFAULT_MODEL = "nvidia/Nemotron-3_5-Lightning"


class NebiusError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProbeDefinition:
    name: str
    description: str
    safety: str = "read_only"


@dataclass(frozen=True)
class ProbeProposal:
    shortlist: tuple[str, ...]
    hypotheses: tuple[dict[str, Any], ...]
    rationale: str
    raw: dict[str, Any]


Transport = Callable[[str, bytes, Mapping[str, str], float], tuple[int, str]]


def _urllib_transport(url: str, body: bytes, headers: Mapping[str, str], timeout: float) -> tuple[int, str]:
    req = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        return exc.code, text


class NemotronClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 45.0,
        max_tokens: int = 192,
        enable_thinking: bool = False,
        transport: Transport = _urllib_transport,
    ):
        self.api_key = api_key or os.getenv("NEBIUS_API_KEY", "")
        self.model = model or os.getenv("NEBIUS_MODEL", DEFAULT_MODEL)
        self.base_url = (base_url or os.getenv("NEBIUS_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.timeout = timeout
        self.max_tokens = int(os.getenv("NEBIUS_MAX_TOKENS", str(max_tokens)))
        env_thinking = os.getenv("NEBIUS_ENABLE_THINKING")
        self.enable_thinking = (env_thinking.strip().lower() in {"1", "true", "yes", "on"}) if env_thinking is not None else enable_thinking
        self.transport = transport
        self.last_call_metrics: dict[str, Any] = {}

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def chat_json(self, system: str, user_payload: Mapping[str, Any], temperature: float = 0.0) -> dict[str, Any]:
        if not self.api_key:
            raise NebiusError("NEBIUS_API_KEY is not configured")
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload, separators=(",", ":"))},
            ],
            "response_format": {"type": "json_object"},
        }
        started = time.perf_counter()
        status, text = self.transport(
            f"{self.base_url}/chat/completions",
            json.dumps(payload).encode("utf-8"),
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            self.timeout,
        )
        latency_s = time.perf_counter() - started
        if status < 200 or status >= 300:
            self.last_call_metrics = {"latency_s": round(latency_s, 3), "http_status": status, "model": self.model}
            raise NebiusError(f"Token Factory HTTP {status}: {text[:300]}")
        try:
            outer = json.loads(text)
            choice = outer["choices"][0]
            content = choice["message"]["content"]
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            parsed = json.loads(content)
            self.last_call_metrics = {
                "latency_s": round(latency_s, 3),
                "http_status": status,
                "model": outer.get("model", self.model),
                "finish_reason": choice.get("finish_reason"),
                "usage": outer.get("usage") or {},
                "enable_thinking": self.enable_thinking,
                "max_tokens": self.max_tokens,
            }
            return parsed
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise NebiusError("Token Factory returned an invalid structured response") from exc


SYSTEM_PROMPT = """You are BREAKPOINT's Probe Architect, a reliability reasoning component.
You receive incident symptoms, current root-service probabilities, recent model-drift state,
and a registry of probes that are already safety-approved. Generate concise hypotheses and
rank only probes from that registry that would most help distinguish the hypotheses.
Never invent shell commands, credentials, destructive actions, or probes outside the registry.
Return JSON with exactly these top-level keys: hypotheses, shortlist, rationale.
- hypotheses: array of objects with service, confidence_hint (0..1), and reason.
- shortlist: array of probe names copied exactly from the registry, most useful first.
- rationale: short explanation of what evidence would distinguish the hypotheses.
Do not claim certainty and do not execute anything."""


class ProbeArchitect:
    def __init__(self, client: NemotronClient):
        self.client = client

    def propose(
        self,
        symptoms: Mapping[str, Any],
        posterior: Mapping[str, float],
        drift_state: Mapping[str, Any],
        probes: Sequence[ProbeDefinition],
        max_probes: int = 3,
    ) -> ProbeProposal:
        registry = {p.name: p for p in probes}
        payload = {
            "symptoms": symptoms,
            "posterior": dict(posterior),
            "drift_state": dict(drift_state),
            "safe_probe_registry": [
                {"name": p.name, "description": p.description, "safety": p.safety}
                for p in probes
            ],
            "max_probes": max_probes,
        }
        raw = self.client.chat_json(SYSTEM_PROMPT, payload)
        incoming = raw.get("shortlist")
        if not isinstance(incoming, list):
            raise NebiusError("Nemotron response did not contain a shortlist array")
        shortlist: list[str] = []
        for value in incoming:
            if isinstance(value, str) and value in registry and value not in shortlist:
                shortlist.append(value)
            if len(shortlist) >= max_probes:
                break
        if not shortlist:
            raise NebiusError("Nemotron did not return any approved probe")
        hypotheses = raw.get("hypotheses", [])
        if not isinstance(hypotheses, list):
            hypotheses = []
        rationale = raw.get("rationale", "")
        if not isinstance(rationale, str):
            rationale = str(rationale)
        return ProbeProposal(tuple(shortlist), tuple(h for h in hypotheses if isinstance(h, dict)), rationale, raw)
