# CIM 3D Scheduler Docs

本项目对应赛道二 / 赛题二：3D 异构 CIM 资源调度优化。V5 提交快照已经完成，推荐先阅读：

- [SUBMISSION_PARAM_CARD_V5.md](SUBMISSION_PARAM_CARD_V5.md)
- `../outputs/final_param_recommendation_v5.json`
- `../outputs/submission_v5_primary/validation_report.json`
- `../outputs/large_scale_v5/run_strict/validation_report.json`

`main` 上已合并 V5 之后的 shared/non-expert operator replication 优化。该优化的严格容量 smoke 产物是 `../outputs/v6_shared_replication_smoke/`，用于后续迭代验证；当前正式提交口径仍以 V5 参数卡和 `submission_v5_*` 输出为准。使用当前 `main` 复现历史 V5 命令时，应显式加入 `--disable-shared-replication`。

## 固定环境

只使用项目虚拟环境：

```powershell
$PY = "E:/Users/Eric/Desktop/Inno/saidao2/.venv/Scripts/python.exe"
```

不要使用裸 `python` 或系统 Python。所有命令默认从 `saidao2/cim_3d_scheduler` 目录运行。

## V5 快速验收

```powershell
& $PY -m pytest -q
& $PY scripts/validate_submission.py --output outputs/submission_v5_primary --strict-capacity --capacity-max-ratio 2.0 --report outputs/submission_v5_primary/validation_report.json
& $PY scripts/validate_submission.py --output outputs/submission_v5_backup --strict-capacity --capacity-max-ratio 2.0 --report outputs/submission_v5_backup/validation_report.json
& $PY scripts/validate_submission.py --output outputs/large_scale_v5/run_strict --strict-capacity --capacity-max-ratio 2.0 --report outputs/large_scale_v5/run_strict/validation_report.json
```

## V5 输出摘要

| 目录 | 用途 | optimized latency | improvement | mapping_rate | capacity_ratio | validator |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `outputs/submission_v5_primary` | 最终主方案 | 234,436 | 56.56% | 1.0 | 1.9862 | pass |
| `outputs/submission_v5_backup` | 最终备选方案 | 258,401 | 52.53% | 1.0 | 1.9862 | pass |
| `outputs/large_scale_v5/run_strict` | 128 experts 大模型证明 | 64,863 | 63.09% | 1.0 | 1.9427 | pass |

## 后续优化 smoke

| 目录 | 用途 | baseline latency | optimized latency | improvement | capacity_ratio | validator |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `outputs/v6_shared_replication_smoke` | shared/non-expert operator replication 严格容量 smoke | 16,706 | 8,073 | 51.68% | 1.9862 | pass |
| `outputs/v6_shared_replication_balanced_candidate` | V5 primary 参数 + shared replication，`trace_balanced` 复核 | 539,729 | 234,436 | 56.56% | 1.9862 | pass |
| `outputs/v6_shared_replication_balanced_budget045` | 放宽共享复制预算探索，`trace_balanced` 复核 | 539,729 | 333,003 | 38.30% | 1.9862 | pass |

`trace_balanced` 上，shared replication 候选没有超过 V5 primary；预算放宽后虽然共享副本实际放置成功，但 latency 退化。因此当前正式提交口径仍以 V5 primary 为准。

## 主要产物

每次 `main.py` 运行会生成：

- `solution.json`: 官方提交风格结果，含顶层 `weight_cubes`、`schedule` 兼容字段。
- `run_manifest.json`: 输入 hash、Cube/MoE/simulator 参数、容量摘要、运行环境。
- `comparison_metrics.json`: baseline/optimized/ablation 对比指标。
- `*_mapping.json`, `*_simulation.json`, `*_solution.json`: 各变体中间产物；当前消融名包含 `ablation_best_fit_replication_only` 和 `ablation_moe_without_replication`。
- `optimized_profile.json`: 推理级 latency、contention 等剖析数据。
- `cooccurrence_heatmap.png`, `optimized_mapping_slices.png`, `optimized_schedule_gantt.png`, `optimized_latency_distribution.png`, `optimized_subcube_contention.png`: 图表。

## V5 代码能力

- 官方容量约束：`--strict-capacity` 和 `--capacity-max-ratio`。
- Cube 参数化：`--cube-n/--cube-d/--cube-h/--cube-w`。
- 官方结构兼容：`solution.json` 顶层新增 `weight_cubes` 与 `schedule`。
- 合规校验：`scripts/validate_submission.py` 检查容量、越界、重叠、Sub-Cube 互斥、依赖违例、unplaced、mapping_rate。
- V5 算法增强：`transition_conflict_weight` 进入 MoE placement 评分，并接入单目标/多目标 Optuna。
- 大模型证明：`scripts/generate_synthetic_model.py` 生成 metadata-only MoE 模型和 trace，不生成真实权重矩阵。

## 当前 main 代码能力

- shared/non-expert operator replication：大共享算子可在额外副本预算内复制，缓解共享层热点。
- CLI 开关：`--disable-shared-replication`、`--shared-replication-min-volume`、`--shared-replication-max-replicas`。
- Mapping 元信息：`optimized_mapping.json` 的 `metadata.shared_operator_replication` 记录启用状态、阈值、请求数和实际副本数。

## 历史说明

V4 参数卡仍保留在 [SUBMISSION_PARAM_CARD_V4.md](SUBMISSION_PARAM_CARD_V4.md)，用于追溯上一轮结果。当前提交口径以 V5 为准。
