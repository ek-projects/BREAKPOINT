"""Hybrid Nemotron + deterministic BREAKPOINT probe selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from active_probe_core import expected_error_reduction
from nebius_reasoner import ProbeArchitect, ProbeDefinition, ProbeProposal


@dataclass(frozen=True)
class HybridDecision:
    selected_probe: str
    expected_error_reduction: float
    architect: ProbeProposal | None
    fallback_used: bool


def choose_next_probe(
    posterior: Mapping[str, float],
    services: Sequence[str],
    likelihoods,
    safe_probes: Sequence[ProbeDefinition],
    architect: ProbeArchitect | None = None,
    symptoms=None,
    drift_state=None,
) -> HybridDecision:
    names = [p.name for p in safe_probes]
    proposal = None
    fallback = False
    candidates = names
    if architect is not None:
        try:
            proposal = architect.propose(
                symptoms or {}, posterior, drift_state or {}, safe_probes,
                max_probes=min(3, len(safe_probes)),
            )
            candidates = list(proposal.shortlist)
        except Exception:
            # Availability or malformed-model-output must never disable diagnosis.
            fallback = True
            candidates = names

    selected = max(
        candidates,
        key=lambda p: (expected_error_reduction(posterior, services, likelihoods, p), p),
    )
    score = expected_error_reduction(posterior, services, likelihoods, selected)
    return HybridDecision(selected, score, proposal, fallback)
