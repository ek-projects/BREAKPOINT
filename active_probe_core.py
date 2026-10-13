"""Reusable active-probing evaluator used by BREAKPOINT external benchmarks."""
from __future__ import annotations

import csv
import io
import itertools
import statistics
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Observation:
    impaired: bool
    insufficient: bool
    magnitude: float


@dataclass
class Case:
    name: str
    service: str
    fault: str
    observations: dict[str, Observation]


def parse_cases(text: str, probes: Sequence[str], allowed_types: set[tuple[str, str]]) -> dict[str, Case]:
    probe_set = set(probes)
    cases: dict[str, Case] = {}
    for row in csv.DictReader(io.StringIO(text)):
        if row.get("journey_label") not in probe_set:
            continue
        incident = (row["target_service"], row["fault_type"])
        if incident not in allowed_types:
            continue
        case = cases.setdefault(
            row["case"], Case(row["case"], row["target_service"], row["fault_type"], {})
        )
        case.observations[row["journey_label"]] = Observation(
            impaired=row["impaired"].strip().lower() == "true",
            insufficient=row["insufficient_data"].strip().lower() == "true",
            magnitude=float(row.get("impairment_magnitude") or 0.0),
        )
    return cases


def positive_median(cases: Iterable[Case], conservative: bool = True) -> float:
    values = [
        o.magnitude
        for c in cases
        for o in c.observations.values()
        if o.impaired and o.magnitude > 0 and not (conservative and o.insufficient)
    ]
    return statistics.median(values) if values else 0.0


def category(o: Observation, threshold: float, conservative: bool = True) -> int | None:
    if conservative and o.insufficient:
        return None
    if not o.impaired:
        return 0
    return 1 if o.magnitude <= threshold else 2


def fit_likelihoods(
    train_cases: Sequence[Case], services: Sequence[str], probes: Sequence[str], alpha: float = 0.5
):
    threshold = positive_median(train_cases, True)
    counts = {(h, p): [alpha, alpha, alpha] for h in services for p in probes}
    for case in train_cases:
        for probe, obs in case.observations.items():
            y = category(obs, threshold, True)
            if y is not None:
                counts[(case.service, probe)][y] += 1
    likelihoods = {k: tuple(v / sum(vals) for v in vals) for k, vals in counts.items()}
    return threshold, likelihoods


def update(posterior, services, likelihoods, probe, outcome):
    weights = {h: posterior[h] * likelihoods[(h, probe)][outcome] for h in services}
    total = sum(weights.values())
    return ({h: weights[h] / total for h in services} if total else dict(posterior))


def expected_error_reduction(posterior, services, likelihoods, probe):
    before = 1.0 - max(posterior.values())
    after = 0.0
    for outcome in range(3):
        py = sum(posterior[h] * likelihoods[(h, probe)][outcome] for h in services)
        if py:
            po = update(posterior, services, likelihoods, probe, outcome)
            after += py * (1.0 - max(po.values()))
    return before - after


def diagnose(case, services, probes, likelihoods, threshold, budget, strategy, order=None):
    posterior = {h: 1.0 / len(services) for h in services}
    used: list[str] = []
    chosen: list[str] = []
    for step in range(budget):
        available = [p for p in probes if p not in used]
        if not available:
            break
        if strategy == "active":
            probe = max(
                available,
                key=lambda p: (expected_error_reduction(posterior, services, likelihoods, p), p),
            )
        elif strategy == "fixed":
            probe = order[step]
        else:
            raise ValueError("strategy must be active or fixed")
        used.append(probe)
        chosen.append(probe)
        obs = case.observations.get(probe)
        if obs is None:
            continue
        outcome = category(obs, threshold, True)
        if outcome is not None:
            posterior = update(posterior, services, likelihoods, probe, outcome)
    prediction = max(sorted(services), key=lambda h: posterior[h])
    return prediction, posterior, tuple(chosen)


def evaluate(
    cases: dict[str, Case], train_types: set[tuple[str, str]], test_types: set[tuple[str, str]],
    probes: Sequence[str], alpha: float = 0.5, budget: int = 3,
):
    train = [c for c in cases.values() if (c.service, c.fault) in train_types]
    test = [c for c in cases.values() if (c.service, c.fault) in test_types]
    services = tuple(sorted({c.service for c in train + test}))
    threshold, likelihoods = fit_likelihoods(train, services, probes, alpha)
    prior = {h: 1.0 / len(services) for h in services}
    static_order = tuple(sorted(
        probes, key=lambda p: (-expected_error_reduction(prior, services, likelihoods, p), p)
    ))

    active_correct: list[int] = []
    static_correct: list[int] = []
    details = []
    for case in test:
        pa, posta, chosen = diagnose(case, services, probes, likelihoods, threshold, budget, "active")
        ps, _, _ = diagnose(case, services, probes, likelihoods, threshold, budget, "fixed", static_order)
        a = int(pa == case.service)
        s = int(ps == case.service)
        active_correct.append(a)
        static_correct.append(s)
        details.append({
            "case": case.name, "service": case.service, "fault": case.fault,
            "active_prediction": pa, "static_prediction": ps,
            "active_correct": bool(a), "static_correct": bool(s),
            "active_probes": list(chosen), "true_posterior": posta[case.service],
        })

    random_case_success = [0.0] * len(test)
    random_order_accuracies = []
    all_orders = list(itertools.permutations(probes))
    for order in all_orders:
        correctness = []
        for i, case in enumerate(test):
            pred, _, _ = diagnose(case, services, probes, likelihoods, threshold, budget, "fixed", order)
            ok = int(pred == case.service)
            correctness.append(ok)
            random_case_success[i] += ok
        random_order_accuracies.append(sum(correctness) / len(test))
    random_case_success = [x / len(all_orders) for x in random_case_success]

    return {
        "n_train_cases": len(train), "n_test_cases": len(test), "services": list(services),
        "threshold": threshold, "static_order": list(static_order),
        "active_accuracy": sum(active_correct) / len(active_correct),
        "static_accuracy": sum(static_correct) / len(static_correct),
        "random_order_mean_accuracy": statistics.mean(random_order_accuracies),
        "random_order_min_accuracy": min(random_order_accuracies),
        "random_order_max_accuracy": max(random_order_accuracies),
        "active_only_correct": sum(a == 1 and s == 0 for a, s in zip(active_correct, static_correct)),
        "static_only_correct": sum(a == 0 and s == 1 for a, s in zip(active_correct, static_correct)),
        "active_correct_vector": active_correct, "static_correct_vector": static_correct,
        "random_case_success": random_case_success, "details": details,
    }
