# Optimization Plus Report

Date: 2026-03-31

## What Was Added

- Scheduler overlap knobs:
  - `overlap_transfer_compute` switch
  - `overlap_alpha` in `[0, 1]` for partial overlap strength
- Scheduler load-balance knob:
  - `load_balance_weight` to avoid overloading busy sub-cubes
- Replica placement diversity:
  - additional replicas now prefer unused sub-cubes first
- Extended search tooling:
  - `scripts/short_term_optimize.py` now sweeps `overlap_alpha` and `load_balance_weight`
  - `scripts/tune_optuna.py` now includes overlap/load-balance variables
  - `scripts/tune_optuna_multi.py` now includes overlap/load-balance variables for Pareto search

## Validation Results

### 1) Expanded short-term grid (24 trials)

- Output: `outputs/short_term_opt_plus/short_term_best.json`
- Best config:
  - `replication_volume_budget_ratio=0.15`
  - `cold_quant_bits=4`
  - `cold_sparsity_ratio=0.3`
  - `overlap_transfer_compute=true`
  - `overlap_alpha=0.7`
  - `load_balance_weight=0.0`
- Best optimized latency: `499804`

### 2) Multi-objective Optuna (8 trials)

- Output: `outputs/tuning_multi_plus/multiobjective_pareto.json`
- Best Pareto point:
  - `latency=479842`
  - `conflict_score=390`
  - `space_utilization=0.1293402777777778`

### 3) Comparison against earlier mock run

- Earlier optimized latency (mock run): `751396`
  - source: `outputs/run_with_mock/comparison_metrics.json`
- New best latency (multi-objective trial): `479842`
  - source: `outputs/tuning_multi_plus/trial_004/comparison_metrics.json`
- Relative reduction from earlier optimized run:
  - `(751396 - 479842) / 751396 = 36.14%`

## Commands Used

```bash
python -m pytest -q
python scripts/short_term_optimize.py --model data/sample_model.json --trace outputs/trace_exporter_mock.json --output outputs/short_term_opt_plus --replica-ratios 0.15,0.25,0.35 --cold-quant-bits 4,5 --cold-sparsity 0.3 --overlap-flags true --overlap-alphas 0.7,1.0 --lb-weights 0.0,0.05
python scripts/tune_optuna_multi.py --model data/sample_model.json --trace outputs/trace_exporter_mock.json --output outputs/tuning_multi_plus --trials 8 --overlap-transfer-compute
python scripts/tune_optuna.py --model data/sample_model.json --trace outputs/trace_exporter_mock.json --output outputs/tuning_plus --trials 4
```
