#!/usr/bin/env python3
"""Frozen confirmatory BREAKPOINT benchmark on RCAEval RE2 Train Ticket.

The configuration was frozen in CONFIRMATORY_FREEZE_2026-10-13.json before
Train Ticket impairment outcomes were evaluated. Do not retune these constants
and still call the same corpus confirmatory.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import urllib.request
from pathlib import Path

from active_probe_core import evaluate, parse_cases

SOURCE_URL = "https://raw.githubusercontent.com/Tanmay268/BLAST/main/results/data/re2tt_journey_impairment_full.csv"
PROBES = (
    "POST /api/v1/users/login (direct)",
    "POST /api/v1/travel2service/trips/left (direct)",
    "POST /api/v1/travelservice/trips/left (direct)",
    "GET /api/v1/foodservice/foods/{date}/{startStation}/{endStation}/{tripId} (direct)",
    "GET /api/v1/adminbasicservice/adminbasic/prices (direct)",
)
TRAIN_TYPES = {
    ("ts-auth-service", "delay"), ("ts-auth-service", "disk"), ("ts-auth-service", "loss"), ("ts-auth-service", "mem"),
    ("ts-order-service", "disk"), ("ts-order-service", "loss"), ("ts-order-service", "mem"), ("ts-order-service", "socket"),
    ("ts-route-service", "cpu"), ("ts-route-service", "delay"), ("ts-route-service", "mem"), ("ts-route-service", "socket"),
    ("ts-train-service", "delay"), ("ts-train-service", "disk"), ("ts-train-service", "loss"), ("ts-train-service", "mem"),
    ("ts-travel-service", "cpu"), ("ts-travel-service", "disk"), ("ts-travel-service", "loss"), ("ts-travel-service", "socket"),
}
TEST_TYPES = {
    ("ts-auth-service", "cpu"), ("ts-auth-service", "socket"),
    ("ts-order-service", "cpu"), ("ts-order-service", "delay"),
    ("ts-route-service", "disk"), ("ts-route-service", "loss"),
    ("ts-train-service", "cpu"), ("ts-train-service", "socket"),
    ("ts-travel-service", "delay"), ("ts-travel-service", "mem"),
}
ALPHA = 0.5
BUDGET = 3


def exact_mcnemar(active, static):
    a_only = sum(a == 1 and s == 0 for a, s in zip(active, static))
    s_only = sum(a == 0 and s == 1 for a, s in zip(active, static))
    n = a_only + s_only
    if not n:
        return 1.0
    k = min(a_only, s_only)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def bootstrap_difference(a, b, draws=20000, seed=20261013):
    rng = random.Random(seed)
    n = len(a)
    values = []
    for _ in range(draws):
        idx = [rng.randrange(n) for _ in range(n)]
        values.append(sum(a[i] - b[i] for i in idx) / n)
    values.sort()
    return [values[int(0.025 * draws)], values[int(0.975 * draws)]]


def read_source(path=None):
    if path:
        return Path(path).read_text(encoding="utf-8")
    with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
        return response.read().decode("utf-8")


def run(text):
    cases = parse_cases(text, PROBES, TRAIN_TYPES | TEST_TYPES)
    result = evaluate(cases, TRAIN_TYPES, TEST_TYPES, PROBES, ALPHA, BUDGET)
    active = result.pop("active_correct_vector")
    static = result.pop("static_correct_vector")
    random_case = result.pop("random_case_success")
    result.update({
        "status": "frozen_confirmatory_run",
        "source": SOURCE_URL,
        "configuration": {
            "mode": "ternary", "conservative": True, "alpha": ALPHA,
            "budget": BUDGET, "probes": list(PROBES),
        },
        "mcnemar_exact_two_sided_p": exact_mcnemar(active, static),
        "bootstrap95_active_minus_static": bootstrap_difference(active, static),
        "active_minus_exact_random_mean": sum(a-r for a, r in zip(active, random_case)) / len(active),
    })
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv")
    p.add_argument("--output")
    a = p.parse_args()
    out = run(read_source(a.csv))
    text = json.dumps(out, indent=2, sort_keys=True)
    print(text)
    if a.output:
        Path(a.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
