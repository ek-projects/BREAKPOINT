"""BREAKPOINT: Bayesian fault diagnosis and budget-aware experiment planning.

Synthetic sandbox only: no production services, network calls or destructive faults.
"""
from dataclasses import dataclass
from math import log2
from random import Random
from typing import Sequence


@dataclass(frozen=True)
class Experiment:
    name: str
    cost: float
    detection: tuple[float, ...]


@dataclass(frozen=True)
class Scenario:
    prior: tuple[float, ...]
    experiments: tuple[Experiment, ...]


def entropy(probabilities: Sequence[float]) -> float:
    return -sum(p * log2(p) for p in probabilities if p > 0)


def validate(scenario: Scenario) -> None:
    n = len(scenario.prior)
    if not n or any(p < 0 for p in scenario.prior) or abs(sum(scenario.prior) - 1) > 1e-8:
        raise ValueError('Prior must be a nonempty normalized distribution')
    if not scenario.experiments:
        raise ValueError('At least one experiment is required')
    for e in scenario.experiments:
        if e.cost <= 0 or len(e.detection) != n or any(d < 0 or d > 1 for d in e.detection):
            raise ValueError(f'Invalid experiment: {e.name}')


def probability_positive(e: Experiment, prior: Sequence[float]) -> float:
    return sum(p * d for p, d in zip(prior, e.detection))


def posterior(e: Experiment, prior: Sequence[float], observed_positive: bool) -> tuple[float, ...]:
    weights = [p * (d if observed_positive else 1-d) for p, d in zip(prior, e.detection)]
    total = sum(weights)
    if total <= 0:
        raise ValueError('Impossible observation under model')
    return tuple(x / total for x in weights)


def information_gain(e: Experiment, prior: Sequence[float]) -> float:
    q = probability_positive(e, prior)
    after = 0.0
    if q > 0:
        after += q * entropy(posterior(e, prior, True))
    if q < 1:
        after += (1-q) * entropy(posterior(e, prior, False))
    return max(0.0, entropy(prior) - after)


def select_experiment(scenario: Scenario, prior: Sequence[float], remaining: float, used: frozenset[int], policy: str, rng: Random) -> int | None:
    available = [i for i, e in enumerate(scenario.experiments) if i not in used and e.cost <= remaining + 1e-9]
    if not available:
        return None
    if policy == 'random':
        return rng.choice(available)
    if policy == 'cheapest':
        return min(available, key=lambda i: (scenario.experiments[i].cost, i))
    if policy == 'information':
        return max(available, key=lambda i: (information_gain(scenario.experiments[i], prior) / scenario.experiments[i].cost, -i))
    raise ValueError('Unknown policy')


def run_trial(scenario: Scenario, truth: int, policy: str, budget: float, seed: int) -> dict:
    validate(scenario)
    rng = Random(seed)
    prior = scenario.prior
    used: frozenset[int] = frozenset()
    spent = 0.0
    history = []
    while True:
        index = select_experiment(scenario, prior, budget-spent, used, policy, rng)
        if index is None:
            break
        e = scenario.experiments[index]
        observation = rng.random() < e.detection[truth]
        prior = posterior(e, prior, observation)
        spent += e.cost
        used = used | {index}
        history.append({'experiment': e.name, 'positive': observation, 'posterior': prior})
    predicted = max(range(len(prior)), key=lambda i: prior[i])
    return {'correct': predicted == truth, 'predicted': predicted, 'truth': truth,
            'spent': spent, 'confidence': max(prior), 'history': history}


def benchmark(scenario: Scenario, policies=('information','random','cheapest'), budget=2.0, trials=5000, seed=42) -> dict:
    validate(scenario)
    if trials < 1 or budget < 0:
        raise ValueError('Invalid budget or trials')
    master = Random(seed)
    truths = master.choices(range(len(scenario.prior)), weights=scenario.prior, k=trials)
    result = {}
    for policy in policies:
        runs = [run_trial(scenario, truth, policy, budget, seed + i * 10007) for i, truth in enumerate(truths)]
        result[policy] = {'accuracy': sum(r['correct'] for r in runs)/trials,
                          'avg_cost': sum(r['spent'] for r in runs)/trials,
                          'trials': trials}
    return result


DEMO = Scenario(
    prior=(.45, .35, .20),
    experiments=(
        Experiment('database latency', 2.0, (.90, .20, .10)),
        Experiment('queue saturation', 1.0, (.10, .85, .25)),
        Experiment('dependency timeout', 1.5, (.20, .20, .95)),
    ),
)
