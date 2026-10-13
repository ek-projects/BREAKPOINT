# BREAKPOINT evaluation methodology

## Evidence tiers

1. **Synthetic mechanics** — validates Bayesian selection, drift detection and online calibration, but does not establish external validity.
2. **Online Boutique external development benchmark** — validates the external data path, but is no longer a pristine holdout because configuration choices were inspected against it.
3. **Train Ticket frozen confirmation** — configuration was written to `CONFIRMATORY_FREEZE_2026-10-13.json` before Train Ticket impairment outcomes were evaluated.

## Frozen Train Ticket protocol

- Source: BLAST-processed RCAEval RE2 Train Ticket journey impairment table.
- Split: published `split_re2tt_v1.yaml`, grouping by `(service, fault_type)`; 60 training and 30 held-out cases.
- Target: root service, not fault type.
- Probe set: five highest-volume journey signatures, selected from trace counts only.
- Observation: ternary — not impaired, impaired-low, impaired-high.
- Low/high threshold: median positive impairment magnitude computed from training cases only.
- Missingness: `insufficient_data=True` is treated as missing.
- Prior smoothing: Jeffreys/Dirichlet alpha 0.5.
- Budget: three probes.
- Active objective: expected reduction in Bayes 0-1 classification error.
- Baselines: learned fixed probe order and exact evaluation of all 5! fixed orders.
- No Train Ticket parameter tuning after observing results.

## Current limitations

- Only 30 held-out Train Ticket cases; exact paired significance is weak (McNemar p=0.125).
- The benchmark replays published telemetry summaries; it is not a live production fault-injection experiment.
- Five candidate services are evaluated, not the full production dependency graph.
- The probabilistic model is simple and does not yet use Nebius/NVIDIA models.
- External comparison against established RCA algorithms is still needed; current baselines test probe scheduling, not end-to-end RCA superiority.

## Claim discipline

Do not claim that BREAKPOINT is a novel Bayesian algorithm, universally beats existing RCA systems, or has production-proven reliability. The current defensible contribution is the closed-loop combination of adaptive probe choice, fast recalibration under drift, explicit uncertainty, and reproducible external evaluation under a strict probe budget.
