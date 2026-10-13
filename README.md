# BREAKPOINT

**Evidence-driven active root-cause diagnosis with Nebius Token Factory and NVIDIA Nemotron.**

BREAKPOINT diagnoses reliability incidents by choosing the **next diagnostic probe that is expected to reduce uncertainty the most**, instead of blindly running a fixed runbook. NVIDIA Nemotron proposes a grounded shortlist from a safety-approved probe registry; a deterministic Bayesian planner keeps final authority over which probe is executed.

## Why this exists

Traditional incident workflows often execute the same checks in the same order even when earlier evidence has already made some checks low-value. BREAKPOINT treats diagnosis as a bounded sequential decision problem:

1. **Probe Architect — NVIDIA Nemotron 3.5 Lightning** converts incident context into hypotheses and ranks only pre-approved probes.
2. **Safety boundary** rejects invented or non-whitelisted probes.
3. **Bayesian planner** chooses the next probe by expected reduction in Bayes classification error.
4. **Experiment runner** executes the selected probe. The public demo uses loopback-only HTTP probes.
5. **Posterior update** revises root-service probabilities from the observation.
6. **Model Skeptic** monitors verified error and calibration drift so stale reliability assumptions can be detected.

The LLM is advisory by design. It cannot execute arbitrary shell commands, create executable probes, or override the deterministic selector.

## Hackathon runtime

- **Provider:** Nebius Token Factory
- **NVIDIA open model:** `nvidia/Nemotron-3_5-Lightning`
- **API:** OpenAI-compatible chat completions
- **Structured mode:** JSON output, `temperature=0`, `max_tokens=192`, `enable_thinking=false`
- **Model-call budget:** one Probe Architect attempt per incident; later probe steps remain deterministic
- **Failure behavior:** malformed output or model/API unavailability falls back immediately to the deterministic planner

The live client records only safe telemetry such as latency, model ID, token usage, finish reason and thinking mode. Credentials are never included in telemetry.

## Interactive demo

The demo is intentionally safe: every diagnostic action targets a temporary service bound to `127.0.0.1`.

```bash
cp .env.example .env
# edit .env and set NEBIUS_API_KEY locally
set -a; . ./.env; set +a
BREAKPOINT_PORT=8878 python3 demo_server.py
```

Then open `http://127.0.0.1:8878`.

Without `NEBIUS_API_KEY`, the same flow runs in deterministic mode. The fault selector is visible for judging transparency, but the diagnosis routine receives only observations produced by probes.

## Live Nebius validation — 13 October 2026

A real Token Factory run with `nvidia/Nemotron-3_5-Lightning` completed all three loopback fault cases correctly with **zero fallback steps**:

| Hidden fault | Prediction | Confidence | Steps | End-to-end latency |
| --- | --- | ---: | ---: | ---: |
| database | database | 96.0% | 2 | 1.491 s |
| queue | queue | 96.0% | 1 | 1.479 s |
| dependency | dependency | 97.9% | 2 | 1.427 s |

The current one-call configuration averages **1.466 s end-to-end**. That is about **70.8% lower latency** than the previous 5.024 s optimized path and about **95.6% lower** than the original 33.295 s path. Each incident used exactly one successful Nemotron call, zero fallback steps, `enable_thinking=false`, and a 192-token completion cap. Observed model-call latency was 0.922-0.987 s with zero reasoning tokens.

This is an **integration validation**, not an accuracy benchmark. Three loopback cases do not establish real-world diagnostic performance. See `live_nebius_onecall_validation_2026-10-13.json`.

## Frozen external confirmatory benchmark

BREAKPOINT was also evaluated on BLAST-processed **RCAEval RE2 Train Ticket** data. The protocol was frozen before inspecting Train Ticket outcomes; see `CONFIRMATORY_FREEZE_2026-10-13.json`.

With a fixed three-probe budget on 30 held-out cases:

| Strategy | Root-service accuracy |
| --- | ---: |
| BREAKPOINT active probing | **73.3% (22/30)** |
| Learned static probe order | 60.0% (18/30) |
| Mean over all 120 fixed orders | 53.7% |
| Best fixed order, selected retrospectively | 70.0% |

The paired active-vs-static difference is +13.3 percentage points. All four discordant cases favor BREAKPOINT, but the exact two-sided McNemar test is **p=0.125** because the holdout is small. A paired bootstrap interval is roughly +3.3 to +26.7 percentage points. The direction is encouraging but is **not definitive statistical proof**.

Crucially, this frozen result validates the adaptive planner. It must **not** be presented as evidence that Nemotron caused the 22/30 result; the hosted Nemotron integration was validated separately.

## Drift and adaptation research

Synthetic stress tests also exercise online calibration and drift detection. They are useful engineering tests, not external-validity evidence. One important negative result is preserved: online adaptation showed faster recovery immediately after abrupt drift, but **no long-run advantage over periodic recalibration** was demonstrated.

## Reproduce

The project uses Python's standard library for the core and demo.

```bash
python -m unittest discover -s tests -v
python confirmatory_tt_benchmark.py   # requires the documented RCAEval/BLAST inputs
```

The reconstructed server working tree currently passes **9 regression tests**. The retained candidate bundle contains the broader historical research suite; only tests aligned with the current one-call runtime are counted here.

## Repository map

- `nebius_reasoner.py` — Token Factory client and safety-bounded Probe Architect
- `hybrid_planner.py` — Nemotron shortlist + deterministic Bayesian selection
- `active_probe_core.py` — posterior updates and expected error reduction
- `model_skeptic.py` — online calibration and drift monitoring
- `demo_runtime.py` — end-to-end diagnosis loop
- `demo_server.py`, `web/` — zero-dependency interactive demo
- `CONFIRMATORY_FREEZE_2026-10-13.json` — preregistered confirmatory configuration
- `external_tt_confirmatory_results.json` — held-out Train Ticket results
- `live_nebius_onecall_validation_2026-10-13.json` — current hosted one-call runtime validation
- `tests/` — unit and integration tests

## Safety

The included demo does not target production systems. Executed demo probes are whitelisted HTTP requests to loopback only. A production adapter would require an explicit probe registry, authorization policy, auditing, rate limits and environment-specific safety controls.

## Evidence boundaries

BREAKPOINT is a research/hackathon prototype. Bayesian active diagnosis, adaptive experiment selection and concept-drift monitoring all have prior art. The contribution demonstrated here is the integration of a safety-bounded LLM Probe Architect with deterministic active probing, explicit fallback behavior, reproducible external evaluation, and a measured hosted runtime path.

See `METHODOLOGY.md` for the experimental protocol.
