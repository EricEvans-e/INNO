# Optimization Data+Algo Report

Date: 2026-03-31

## Data-layer Enhancements

- Added trace feature extraction in parser:
  - `transition_matrix`
  - `expert_burstiness`
  - `expert_transition_influence`
- Added trace scenario generator:
  - `scripts/generate_trace_variants.py`
  - outputs base/hotspot/balanced/bursty variants for robust tuning.
- Upgraded short-term optimizer to support multi-trace robust objective:
  - average metrics + worst-latency penalty (`robust_worst_weight`).

## Algorithm-layer Enhancements

- Adaptive replica pressure now combines:
  - frequency
  - contention
  - burstiness
  - transition influence
- New MoEConfig knobs:
  - `replication_freq_weight`
  - `replication_contention_weight`
  - `replication_burst_weight`
  - `replication_transition_weight`

## Validation

- Unit tests: 2026-04-29 本地验证为 `5 passed`，另有 1 个 Matplotlib `get_cmap` 弃用 warning
- Robust optimization on 3 trace scenarios (`base/hotspot/bursty`), 4 trials:
  - output: `outputs/short_term_opt_robust/short_term_best.json`
  - best:
    - `latency(mean)=383266.67`
    - `worst_latency=499804.0`
    - `conflict_score(mean)=209.0`
    - `space_utilization(mean)=0.1328125`
    - `temporal_utilization(mean)=0.4715`

- Generalization check on held-out balanced trace:
  - output: `outputs/run_robust_on_balanced/comparison_metrics.json`
  - optimized latency: `422970`
  - baseline latency: `2057266`
  - latency improvement: `79.44%`
