# Submission Parameter Card V5

## 1. 定位

本参数卡面向赛道二 / 赛题二：3D 异构 CIM 资源调度优化。V5 在 V4 基础上补齐官方合规约束、validator、严格容量复现、大模型 metadata-only 适配证明，并新增 transition-aware placement penalty。

固定 Python 环境：

```powershell
$PY = "E:/Users/Eric/Desktop/Inno/saidao2/.venv/Scripts/python.exe"
```

不要使用系统 Python 或裸 `python` 命令。

2026-05-22 之后的 `main` 默认包含 post-V5 shared/non-expert operator replication。为了严格复现 V5 历史基线，本参数卡中的 `main.py` 复现命令显式加入 `--disable-shared-replication`。

## 2. V5 主方案

来源：`outputs/tuning_v5_strict_multi_venv/multiobjective_pareto.json` rank 1 / trial 5。

输出目录：`outputs/submission_v5_primary`

留出集 `trace_balanced` 复跑结果：

- baseline latency: 539,729
- optimized latency: 234,436
- latency improvement: 56.56%
- conflict_score: 2.0
- space_utilization: 0.5868
- temporal_utilization: 0.5431
- mapping_rate: 1.0
- capacity_ratio: 1.9862 <= 2.0
- validator: pass

关键参数：

- Cube: `N=3, D=2, H=W=4096, max_parallel_subcubes=9`
- overlap: `nonlinear_bandwidth_aware`, `alpha=0.7872434125774002`
- local search: `restarts=5`, `iters=45`, `parallel_trials=12`
- SA: `steps=158`, `init_temp=0.8412572507437754`, `cooling=0.9844704719810294`
- compression: `cold_quant_bits=4`, `cold_sparsity=0.42304229294971546`, `hot_sparsity=0.1616279202918481`
- placement: `conflict=0.711261150867198`, `transition=0.3736175723130549`, `load=0.4403704597393854`

复现命令：

```powershell
& $PY main.py --model data/sample_model.json --trace outputs/trace_variants_mock/trace_balanced.json --output outputs/submission_v5_primary --profile --cube-d 2 --max-parallel-subcubes 9 --strict-capacity --capacity-max-ratio 2.0 --overlap-transfer-compute --overlap-alpha 0.7872434125774002 --overlap-model-mode nonlinear_bandwidth_aware --overlap-bw-power-law-alpha 1.190361785167963 --overlap-z-depth-penalty 0.028146737699334498 --load-balance-weight 0.09101247347113178 --dispatch-policy fifo --criticality-weight 0.7090779988448124 --resource-pressure-weight 0.20062079820191317 --replica-pressure-low-threshold 0.3564781712566482 --replica-pressure-high-threshold 0.8263651577053367 --disable-dynamic-hot-subgraph-topk --dynamic-hot-subgraph-min-ratio 0.2160811746593056 --dynamic-hot-subgraph-max-ratio 0.39290293138967364 --local-restarts 5 --local-iters 45 --search-trials 12 --grouping-multi-start-trials 9 --hot-subgraph-topk 6 --replication-volume-budget-ratio 0.19281327549469593 --disable-shared-replication --sa-steps 158 --sa-init-temp 0.8412572507437754 --sa-cooling 0.9844704719810294 --cold-quant-bits 4 --cold-sparsity-ratio 0.42304229294971546 --hot-sparsity-ratio 0.1616279202918481 --placement-conflict-weight 0.711261150867198 --transition-conflict-weight 0.3736175723130549 --placement-load-weight 0.4403704597393854 --placement-group-penalty 0.49384264035329484 --replica-diversity-penalty 0.28652115975568737 --disable-aspect-aware-packing --aspect-aware-weight 0.7826551872821396 --fragmentation-penalty-weight 0.16421122534206772 --conflict-propagation-weight 0.8109061798032725 --capacity-peak-weight 0.24692215855959965 --seed 2026
& $PY scripts/validate_submission.py --output outputs/submission_v5_primary --strict-capacity --capacity-max-ratio 2.0 --report outputs/submission_v5_primary/validation_report.json
```

## 3. V5 备选方案

来源：`outputs/tuning_v5_strict_single_venv/best_params.json`。

输出目录：`outputs/submission_v5_backup`

留出集 `trace_balanced` 复跑结果：

- baseline latency: 544,342
- optimized latency: 258,401
- latency improvement: 52.53%
- conflict_score: 5.0
- space_utilization: 0.6146
- temporal_utilization: 0.4980
- mapping_rate: 1.0
- capacity_ratio: 1.9862 <= 2.0
- validator: pass

复现命令：

```powershell
& $PY main.py --model data/sample_model.json --trace outputs/trace_variants_mock/trace_balanced.json --output outputs/submission_v5_backup --profile --cube-d 2 --max-parallel-subcubes 9 --strict-capacity --capacity-max-ratio 2.0 --overlap-transfer-compute --overlap-alpha 0.5977199268412985 --overlap-model-mode nonlinear_bandwidth_aware --overlap-bw-power-law-alpha 0.8327461240498177 --overlap-z-depth-penalty 0.3068248861089168 --load-balance-weight 0.053825499746849616 --dispatch-policy criticality --criticality-weight 0.036108853278776123 --resource-pressure-weight 0.32248753384042755 --replica-pressure-low-threshold 0.6429398765527842 --replica-pressure-high-threshold 0.8866649728180842 --disable-dynamic-hot-subgraph-topk --dynamic-hot-subgraph-min-ratio 0.12185841470151416 --dynamic-hot-subgraph-max-ratio 0.3071221599074529 --local-restarts 6 --local-iters 34 --search-trials 10 --grouping-multi-start-trials 6 --hot-subgraph-topk 6 --replication-volume-budget-ratio 0.2575969015586806 --disable-shared-replication --sa-steps 109 --sa-init-temp 1.9074671229270022 --sa-cooling 0.927011668135846 --cold-quant-bits 4 --cold-sparsity-ratio 0.4191280485211647 --hot-sparsity-ratio 0.22854485611710185 --placement-conflict-weight 0.5046258040922602 --transition-conflict-weight 0.24296618805610876 --placement-load-weight 0.8939954421585856 --placement-group-penalty 0.14663247056153716 --replica-diversity-penalty 0.46268536054753584 --disable-aspect-aware-packing --aspect-aware-weight 0.34067458186529737 --fragmentation-penalty-weight 0.31499285949985967 --conflict-propagation-weight 0.7930592573969756 --capacity-peak-weight 0.4668489939714203 --seed 2026
& $PY scripts/validate_submission.py --output outputs/submission_v5_backup --strict-capacity --capacity-max-ratio 2.0 --report outputs/submission_v5_backup/validation_report.json
```

## 4. 大模型适配证明

V5 生成 DeepSeek/MoE 风格 metadata-only 模型，不生成真实权重矩阵，用于证明大规模专家数下 parser、mapping、schedule、validator 都能跑通。

输出目录：`outputs/large_scale_v5/run_strict`

- experts: 128
- inferences: 256
- top_k: 4
- Cube: `N=3, D=8, H=W=4096`
- capacity_ratio: 1.9427 <= 2.0
- optimized latency: 64,863
- latency improvement: 63.09%
- mapping_rate: 1.0
- validator: pass

复现命令：

```powershell
& $PY scripts/generate_synthetic_model.py --model-output outputs/large_scale_v5/synthetic_model.json --trace-output outputs/large_scale_v5/synthetic_trace.json --num-experts 128 --n-inferences 256 --top-k 4 --seed 2026
& $PY main.py --model outputs/large_scale_v5/synthetic_model.json --trace outputs/large_scale_v5/synthetic_trace.json --output outputs/large_scale_v5/run_strict --profile --cube-d 8 --max-parallel-subcubes 9 --strict-capacity --capacity-max-ratio 2.0 --overlap-transfer-compute --overlap-alpha 0.7872434125774002 --overlap-model-mode nonlinear_bandwidth_aware --overlap-bw-power-law-alpha 1.190361785167963 --overlap-z-depth-penalty 0.028146737699334498 --load-balance-weight 0.09101247347113178 --search-trials 2 --parallel-workers 1 --local-restarts 2 --local-iters 15 --grouping-multi-start-trials 3 --hot-subgraph-topk 6 --replication-volume-budget-ratio 0.10 --disable-shared-replication --sa-steps 30 --cold-quant-bits 4 --cold-sparsity-ratio 0.35 --hot-sparsity-ratio 0.15 --placement-conflict-weight 0.711261150867198 --transition-conflict-weight 0.3736175723130549 --placement-load-weight 0.4403704597393854 --placement-group-penalty 0.49384264035329484 --replica-diversity-penalty 0.28652115975568737 --disable-aspect-aware-packing --fragmentation-penalty-weight 0.16421122534206772 --conflict-propagation-weight 0.8109061798032725 --capacity-peak-weight 0.24692215855959965 --deterministic --seed 2026
& $PY scripts/validate_submission.py --output outputs/large_scale_v5/run_strict --strict-capacity --capacity-max-ratio 2.0 --report outputs/large_scale_v5/run_strict/validation_report.json
```

## 5. 调参命令

```powershell
& $PY scripts/tune_optuna.py --model data/sample_model.json --traces outputs/trace_variants_mock/trace_base.json,outputs/trace_variants_mock/trace_hotspot.json,outputs/trace_variants_mock/trace_bursty.json --holdout-traces outputs/trace_variants_mock/trace_balanced.json --output outputs/tuning_v5_strict_single_venv --trials 32 --n-jobs 4 --enable-two-stage-tuning --auto-trace-weight --robust-worst-weight 0.2 --tail-p95-weight 0.08 --tail-p99-weight 0.12 --cube-d 2 --max-parallel-subcubes 9 --strict-capacity --capacity-max-ratio 2.0 --seed 2026
& $PY scripts/tune_optuna_multi.py --model data/sample_model.json --traces outputs/trace_variants_mock/trace_base.json,outputs/trace_variants_mock/trace_hotspot.json,outputs/trace_variants_mock/trace_bursty.json --holdout-traces outputs/trace_variants_mock/trace_balanced.json --holdout-topk 3 --output outputs/tuning_v5_strict_multi_venv --trials 16 --n-jobs 4 --enable-two-stage-tuning --auto-trace-weight --overlap-transfer-compute --robust-worst-weight 0.2 --tail-p95-weight 0.08 --tail-p99-weight 0.12 --cube-d 2 --max-parallel-subcubes 9 --strict-capacity --capacity-max-ratio 2.0 --seed 2026
```

## 6. 验证状态

- `$PY -m py_compile main.py config/moe_config.py src/mapping_solver.py scripts/tune_optuna.py scripts/tune_optuna_multi.py scripts/validate_submission.py scripts/generate_synthetic_model.py`: pass
- `$PY -m pytest -q`: `9 passed, 17 warnings`
- `outputs/v5_smoke_strict_venv`: validator pass
- `outputs/submission_v5_primary`: validator pass
- `outputs/submission_v5_backup`: validator pass
- `outputs/large_scale_v5/run_strict`: validator pass
