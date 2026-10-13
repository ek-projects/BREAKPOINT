"""Safe end-to-end BREAKPOINT demo: Nemotron reasoning + loopback probes."""
from __future__ import annotations

from typing import Any

from active_probe_core import update
from http_lab import FAULTS, PROBES, local_lab, probe
from hybrid_planner import choose_next_probe
from nebius_reasoner import NemotronClient, ProbeArchitect, ProbeDefinition

SERVICES = tuple(FAULTS)
SAFE_PROBES = tuple(
    ProbeDefinition(
        name=p,
        description={
            "db_probe": "Read-only database health check; distinguishes database failure from upstream failures.",
            "queue_probe": "Read-only queue pressure check; distinguishes queue saturation.",
            "dependency_probe": "Read-only upstream dependency check; distinguishes external dependency timeout.",
        }[p],
    )
    for p in PROBES
)

# Three-outcome likelihoods: 0=healthy, 1=degraded, 2=unknown.
# Values are deliberately not 0/1 so the posterior remains calibrated and explainable.
LIKELIHOODS = {}
for service in SERVICES:
    for p in PROBES:
        target = p.removesuffix("_probe")
        if target == "db":
            target = "database"
        if service == target:
            LIKELIHOODS[(service, p)] = (0.02, 0.97, 0.01)
        else:
            LIKELIHOODS[(service, p)] = (0.97, 0.02, 0.01)


def run_demo(
    fault: str,
    client: NemotronClient | None = None,
    budget: int = 3,
    max_architect_calls: int = 1,
) -> dict[str, Any]:
    if fault not in FAULTS:
        raise ValueError("unknown demo fault")
    posterior = {s: 1.0 / len(SERVICES) for s in SERVICES}
    architect = ProbeArchitect(client) if client and client.configured else None
    used: set[str] = set()
    steps = []
    architect_calls = 0
    architect_attempts = 0

    with local_lab(fault) as url:
        for step_no in range(1, budget + 1):
            available = [p for p in SAFE_PROBES if p.name not in used]
            if not available:
                break
            symptoms = {
                "incident": "HTTP service degradation",
                "observations_so_far": [
                    {"probe": s["probe"], "http_status": s["http_status"]} for s in steps
                ],
            }
            step_architect = architect if architect_attempts < max_architect_calls else None
            if step_architect is not None:
                architect_attempts += 1
            decision = choose_next_probe(
                posterior,
                SERVICES,
                LIKELIHOODS,
                available,
                architect=step_architect,
                symptoms=symptoms,
                drift_state={"status": "stable", "verified_error_alarm": False},
            )
            if decision.architect is not None:
                architect_calls += 1
            status = probe(url, decision.selected_probe)
            outcome = 1 if status != 200 else 0
            posterior = update(posterior, SERVICES, LIKELIHOODS, decision.selected_probe, outcome)
            used.add(decision.selected_probe)
            steps.append({
                "step": step_no,
                "probe": decision.selected_probe,
                "http_status": status,
                "expected_error_reduction": decision.expected_error_reduction,
                "posterior": dict(posterior),
                "nemotron_shortlist": list(decision.architect.shortlist) if decision.architect else [],
                "nemotron_rationale": decision.architect.rationale if decision.architect else "",
                "nemotron_metrics": dict(client.last_call_metrics) if decision.architect and client else {},
                "nemotron_invoked": decision.architect is not None,
                "fallback_used": decision.fallback_used,
            })
            if max(posterior.values()) >= 0.85:
                break

    prediction = max(sorted(SERVICES), key=lambda s: posterior[s])
    return {
        "truth": fault,
        "prediction": prediction,
        "correct": prediction == fault,
        "confidence": posterior[prediction],
        "posterior": posterior,
        "steps": steps,
        "nebius_configured": bool(client and client.configured),
        "model": client.model if client else None,
        "architect_calls": architect_calls,
        "architect_attempts": architect_attempts,
        "max_architect_calls": max_architect_calls,
        "safety": "All executed probes are whitelist-only HTTP requests to a loopback lab.",
    }
