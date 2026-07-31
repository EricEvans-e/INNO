# Submission Parameter Card V5.1 Candidate

## 1. 定位

V5.1 是在 V5 primary 基础上的低风险参数候选：不改模型、parser、mapping/simulator 语义，也不启用 post-V5 shared/non-expert operator replication；核心变化是把专家副本预算从 V5 primary 的 `0.19281327549469593` 提高到 `0.35`，让额外容量优先用于 MoE expert 副本。

固定 Python 环境：

```powershell
$PY = "E:/Users/Eric/Desktop/Inno/saidao2/.venv/Scripts/python.exe"
```

不要使用系统 Python 或裸 `python` 命令。

## 2. 与 V5 Primary 对比

| 目录 | optimized latency | improvement | conflict_score | temporal_utilization | space_utilization | mapping_rate | validator |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `outputs/submission_v5_primary` | 234,436 | 56.56% | 2.0 | 0.5431 | 0.5868 | 1.0 | pass |
| `outputs/submission_v5_1_candidate` | 231,672 | 57.08% | 4.0 | 0.5497 | 0.6701 | 1.0 | pass |

V5.1 相对 V5 primary 降低 `2,764` cycles，约 `1.18%`。代价是空间利用率更高、conflict_score 从 `2.0` 升到 `4.0`，但 temporal utilization、pipeline bubble 和平均带宽利用率略有改善。

容量状态：

- `model_logical_volume`: 152,043,520
- `cube_total_volume`: 301,989,888
- `capacity_ratio`: 1.986206896551724 <= 2.0
- `mapping_rate`: 1.0
- validator: `ok=true`, `errors=[]`

## 3. 关键参数变化

保持 V5 primary 的搜索、overlap、placement 参数，修改：

- `--replication-volume-budget-ratio 0.35`
- `--disable-shared-replication`

候选输出中的复制情况：

- physical placements: 34
- logical Weight-Cubes: 22
- replicated logical expert cubes: 12
- `extra_replica_budget`: 53,215,232
- `extra_replica_used`: 50,331,648
- shared replication: disabled, no extra shared physical replicas

## 4. 复现命令

```powershell
& $PY main.py --model data/sample_model.json --trace outputs/trace_variants_mock/trace_balanced.json --output outputs/submission_v5_1_candidate --profile --cube-d 2 --max-parallel-subcubes 9 --strict-capacity --capacity-max-ratio 2.0 --overlap-transfer-compute --overlap-alpha 0.7872434125774002 --overlap-model-mode nonlinear_bandwidth_aware --overlap-bw-power-law-alpha 1.190361785167963 --overlap-z-depth-penalty 0.028146737699334498 --load-balance-weight 0.09101247347113178 --dispatch-policy fifo --criticality-weight 0.7090779988448124 --resource-pressure-weight 0.20062079820191317 --replica-pressure-low-threshold 0.3564781712566482 --replica-pressure-high-threshold 0.8263651577053367 --disable-dynamic-hot-subgraph-topk --dynamic-hot-subgraph-min-ratio 0.2160811746593056 --dynamic-hot-subgraph-max-ratio 0.39290293138967364 --local-restarts 5 --local-iters 45 --search-trials 12 --grouping-multi-start-trials 9 --hot-subgraph-topk 6 --replication-volume-budget-ratio 0.35 --disable-shared-replication --sa-steps 158 --sa-init-temp 0.8412572507437754 --sa-cooling 0.9844704719810294 --cold-quant-bits 4 --cold-sparsity-ratio 0.42304229294971546 --hot-sparsity-ratio 0.1616279202918481 --placement-conflict-weight 0.711261150867198 --transition-conflict-weight 0.3736175723130549 --placement-load-weight 0.4403704597393854 --placement-group-penalty 0.49384264035329484 --replica-diversity-penalty 0.28652115975568737 --disable-aspect-aware-packing --aspect-aware-weight 0.7826551872821396 --fragmentation-penalty-weight 0.16421122534206772 --conflict-propagation-weight 0.8109061798032725 --capacity-peak-weight 0.24692215855959965 --seed 2026
& $PY scripts/validate_submission.py --output outputs/submission_v5_1_candidate --strict-capacity --capacity-max-ratio 2.0 --report outputs/submission_v5_1_candidate/validation_report.json
```

## 5. 推荐口径

若提交前时间有限，可把 `outputs/submission_v5_1_candidate` 作为 V5 primary 的增强候选，因为它在 `trace_balanced` 上严格容量通过并取得更低 latency。

若还有调参时间，建议继续运行 balanced-targeted 多目标调参，目标是找到 `<231,672` 且 conflict_score 不高于 4 的候选。调参脚本现在支持 `--disable-shared-replication`，应显式加入该开关，确保搜索口径和 V5.1 候选一致。

2026-06-09 已做一轮小规模单进程 sanity search：

- `outputs/tuning_v5_1_balanced_targeted_8_single`
- `--trials 8 --n-jobs 1 --disable-shared-replication`
- 最佳 holdout `trace_balanced` latency 为 `373,249`，未超过 V5 primary/V5.1，不晋级为候选
- 该结果说明小样本盲搜不稳定；正式扩搜应以手动 replay + strict validator 作为晋级依据

```powershell
& $PY scripts/tune_optuna_multi.py --model data/sample_model.json --traces outputs/trace_variants_mock/trace_base.json,outputs/trace_variants_mock/trace_hotspot.json,outputs/trace_variants_mock/trace_bursty.json,outputs/trace_variants_mock/trace_balanced.json --holdout-traces outputs/trace_variants_mock/trace_balanced.json --holdout-topk 5 --output outputs/tuning_v5_1_balanced_targeted_64 --trials 64 --n-jobs 1 --enable-two-stage-tuning --auto-trace-weight --overlap-transfer-compute --robust-worst-weight 0.2 --tail-p95-weight 0.08 --tail-p99-weight 0.12 --cube-d 2 --max-parallel-subcubes 9 --strict-capacity --capacity-max-ratio 2.0 --disable-shared-replication --seed 2026
```

候选晋级规则：

- 先查看 `multiobjective_pareto.json` 中 `holdout_eval[].holdout_metrics.per_trace.trace_balanced.latency`
- 只有低于 `231,672` 的候选才值得手动 replay
- 手动 replay 必须保留 `--disable-shared-replication`
- replay 后运行 `scripts/validate_submission.py --strict-capacity --capacity-max-ratio 2.0`
- validator 需满足 `ok=true`、`errors=[]`、`capacity_ratio<=2.0`、`mapping_rate=1.0`

官方 trace 到达后，应重新执行 parser smoke、candidate replay、validator 和 holdout 对比，再决定是否把 V5.1 升为最终提交。
