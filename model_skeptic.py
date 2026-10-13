"""Adaptive calibration and drift monitoring for BREAKPOINT.

Synthetic research code only. The adaptive model updates from post-incident root-cause
labels and from observations of probes that were actually executed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from engine import Experiment, Scenario


class AdaptiveFaultModel:
    """Bayesian online calibration of fault priors and probe reliability."""

    def __init__(self, scenario: Scenario, detection_strength: float = 8.0, prior_strength: float = 10.0):
        if detection_strength <= 0 or prior_strength <= 0:
            raise ValueError("strengths must be positive")
        self._base = scenario
        self._prior_counts = [max(1e-9, p * prior_strength) for p in scenario.prior]
        self._alpha = []
        self._beta = []
        for experiment in scenario.experiments:
            self._alpha.append([max(1e-9, p * detection_strength) for p in experiment.detection])
            self._beta.append([max(1e-9, (1 - p) * detection_strength) for p in experiment.detection])

    def scenario(self) -> Scenario:
        total = sum(self._prior_counts)
        prior = tuple(x / total for x in self._prior_counts)
        experiments = []
        for i, base in enumerate(self._base.experiments):
            detection = tuple(
                self._alpha[i][j] / (self._alpha[i][j] + self._beta[i][j])
                for j in range(len(prior))
            )
            experiments.append(Experiment(base.name, base.cost, detection))
        return Scenario(prior, tuple(experiments))

    def update(self, truth: int, used: Iterable[int], outcomes: tuple[bool, ...]) -> None:
        if truth < 0 or truth >= len(self._prior_counts):
            raise ValueError("truth index out of range")
        self._prior_counts[truth] += 1.0
        for i in used:
            if i < 0 or i >= len(self._alpha):
                raise ValueError("experiment index out of range")
            if outcomes[i]:
                self._alpha[i][truth] += 1.0
            else:
                self._beta[i][truth] += 1.0


@dataclass
class DriftMonitor:
    """Postmortem performance-drift alarm with an exact binomial control limit."""

    warmup: int = 400
    window: int = 20
    alpha: float = 0.001

    def __post_init__(self):
        if self.warmup < self.window or self.window < 2 or not (0 < self.alpha < 1):
            raise ValueError("invalid drift monitor parameters")
        self.correct: list[int] = []
        self.baseline_error: float | None = None
        self.max_errors_in_window: int | None = None
        self.first_alarm_index: int | None = None

    @staticmethod
    def _binomial_upper_tail(n: int, p: float, k: int) -> float:
        from math import comb
        return sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k, n + 1))

    def _calibrate(self) -> None:
        errors = self.warmup - sum(self.correct[:self.warmup])
        self.baseline_error = (errors + 0.5) / (self.warmup + 1.0)
        threshold = self.window + 1
        for k in range(1, self.window + 1):
            if self._binomial_upper_tail(self.window, self.baseline_error, k) <= self.alpha:
                threshold = k
                break
        self.max_errors_in_window = threshold

    def observe(self, correct: bool) -> bool:
        self.correct.append(int(bool(correct)))
        index = len(self.correct) - 1
        if len(self.correct) == self.warmup:
            self._calibrate()
            return False
        if self.max_errors_in_window is None or len(self.correct) < self.window:
            return False
        errors = self.window - sum(self.correct[-self.window:])
        alarm = errors >= self.max_errors_in_window
        if alarm and self.first_alarm_index is None:
            self.first_alarm_index = index
        return alarm
