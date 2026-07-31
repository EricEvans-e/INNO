# cim_3d_scheduler

## Python 环境

本项目固定使用项目虚拟环境，不使用电脑上的系统 Python：

```powershell
$PY = "E:/Users/Eric/Desktop/Inno/saidao2/.venv/Scripts/python.exe"
```

如需补依赖：

```powershell
& $PY -m pip install -r requirements.txt
```

## 快速验证

```powershell
& $PY -m pytest -q
& $PY main.py --model data/sample_model.json --trace data/sample_trace.json --output outputs/v6_shared_replication_smoke --profile --deterministic --search-trials 1 --parallel-workers 1 --local-restarts 1 --local-iters 5 --disable-sa --disable-parallel-search --cube-d 2 --strict-capacity --capacity-max-ratio 2.0 --replication-volume-budget-ratio 0.45 --shared-replication-max-replicas 2 --shared-replication-min-volume 16777216 --overlap-transfer-compute --overlap-alpha 0.7872434125774002 --overlap-model-mode nonlinear_bandwidth_aware --overlap-bw-power-law-alpha 1.190361785167963 --overlap-z-depth-penalty 0.028146737699334498
& $PY scripts/validate_submission.py --output outputs/v6_shared_replication_smoke --strict-capacity --capacity-max-ratio 2.0 --report outputs/v6_shared_replication_smoke/validation_report.json
```

## 通关指南对齐

赛题二项目采用轻量 JSON internal IR 对齐通关指南中的“模型导入 - 中间表示 - 硬件映射 - 静态调度 - 工程验证”叙事，但不实现赛题一式 MLIR Dialect 或 ISA 后端。详细说明见 [docs/GUIDE_ALIGNMENT.md](docs/GUIDE_ALIGNMENT.md)。

```powershell
& $PY main.py --model data/sample_model.json --trace data/sample_trace.json --output outputs/guide_alignment_ir_smoke --profile --deterministic --search-trials 1 --parallel-workers 1 --local-restarts 1 --local-iters 5 --disable-sa --disable-parallel-search --cube-d 2 --strict-capacity --capacity-max-ratio 2.0 --export-internal-ir
& $PY scripts/validate_submission.py --output outputs/guide_alignment_ir_smoke --strict-capacity --capacity-max-ratio 2.0 --report outputs/guide_alignment_ir_smoke/validation_report.json
```

运行后重点查看 `outputs/guide_alignment_ir_smoke/internal_ir.json`。

## 当前 main 后续优化

V5 仍是当前提交基线；`main` 上已经合并后续 shared/non-expert operator replication 优化，用于继续压低严格容量 smoke 时延。
使用当前 `main` 复现历史 V5 参数卡时，`main.py` 命令需要显式加入 `--disable-shared-replication`，避免跑成 post-V5 行为。

`outputs/v6_shared_replication_smoke/`：

- baseline latency: 16,706
- optimized latency: 8,073
- latency improvement: 51.68%
- conflict_score: 24.0
- mapping_rate: 1.0
- capacity_ratio: 1.9862 <= 2.0
- validator: pass

`trace_balanced` 复核结论：

- `outputs/v6_shared_replication_balanced_candidate`: latency 234,436，等同 V5 primary；shared replication 请求 5 个逻辑块但实际额外物理副本为 0。
- `outputs/v6_shared_replication_balanced_budget045`: latency 333,003，validator pass，但低于 V5 primary；共享副本实际生效后挤占/扰动专家副本收益。
- 因此当前正式提交仍以 V5 primary 为准，暂不生成 V6 参数卡。

## V5 最终结果

主方案 `outputs/submission_v5_primary/`：

- optimized latency: 234,436
- latency improvement: 56.56%
- conflict_score: 2.0
- mapping_rate: 1.0
- capacity_ratio: 1.9862 <= 2.0
- validator: pass

V5.1 增强候选 `outputs/submission_v5_1_candidate/`：

- optimized latency: 231,672
- latency improvement: 57.08%
- conflict_score: 4.0
- mapping_rate: 1.0
- capacity_ratio: 1.9862 <= 2.0
- validator: pass
- 核心变化：`--replication-volume-budget-ratio 0.35` + `--disable-shared-replication`

备选方案 `outputs/submission_v5_backup/`：

- optimized latency: 258,401
- latency improvement: 52.53%
- conflict_score: 5.0
- mapping_rate: 1.0
- capacity_ratio: 1.9862 <= 2.0
- validator: pass

大模型证明 `outputs/large_scale_v5/run_strict/`：

- synthetic metadata-only MoE: 128 experts, 256 inferences, top_k 4
- Cube: `N=3, D=8, H=W=4096`
- optimized latency: 64,863
- latency improvement: 63.09%
- mapping_rate: 1.0
- capacity_ratio: 1.9427 <= 2.0
- validator: pass

## V5 调参命令

以下命令用于复查历史 V5 调参入口。V5.1 之后如果要继续围绕 `submission_v5_1_candidate` 扩搜，应按 [docs/SUBMISSION_PARAM_CARD_V5_1.md](docs/SUBMISSION_PARAM_CARD_V5_1.md) 使用 `--disable-shared-replication`；调参脚本现在会校验必须由项目 `.venv` Python 启动。

```powershell
& $PY scripts/tune_optuna.py --model data/sample_model.json --traces outputs/trace_variants_mock/trace_base.json,outputs/trace_variants_mock/trace_hotspot.json,outputs/trace_variants_mock/trace_bursty.json --holdout-traces outputs/trace_variants_mock/trace_balanced.json --output outputs/tuning_v5_strict_single_venv --trials 32 --n-jobs 4 --enable-two-stage-tuning --auto-trace-weight --robust-worst-weight 0.2 --tail-p95-weight 0.08 --tail-p99-weight 0.12 --cube-d 2 --max-parallel-subcubes 9 --strict-capacity --capacity-max-ratio 2.0 --disable-shared-replication --seed 2026
& $PY scripts/tune_optuna_multi.py --model data/sample_model.json --traces outputs/trace_variants_mock/trace_base.json,outputs/trace_variants_mock/trace_hotspot.json,outputs/trace_variants_mock/trace_bursty.json --holdout-traces outputs/trace_variants_mock/trace_balanced.json --holdout-topk 3 --output outputs/tuning_v5_strict_multi_venv --trials 16 --n-jobs 4 --enable-two-stage-tuning --auto-trace-weight --overlap-transfer-compute --robust-worst-weight 0.2 --tail-p95-weight 0.08 --tail-p99-weight 0.12 --cube-d 2 --max-parallel-subcubes 9 --strict-capacity --capacity-max-ratio 2.0 --disable-shared-replication --seed 2026
```

完整 primary/backup/large-scale 复现命令见 [docs/SUBMISSION_PARAM_CARD_V5.md](docs/SUBMISSION_PARAM_CARD_V5.md)，V5.1 增强候选见 [docs/SUBMISSION_PARAM_CARD_V5_1.md](docs/SUBMISSION_PARAM_CARD_V5_1.md)。

## 项目结构

```text
cim_3d_scheduler/
├── config/
├── data/
├── docs/
├── outputs/
├── scripts/
├── src/
├── tests/
├── main.py
├── requirements.txt
└── environment.yml
```

## V5 关键增强

- `main.py` 支持 `--cube-n/--cube-d/--cube-h/--cube-w`、`--capacity-max-ratio`、`--strict-capacity`。
- `solution.json` 顶层兼容 `weight_cubes` 与 `schedule`，同时保留原 `spatial_mapping` 和 `static_schedule`。
- `scripts/validate_submission.py` 校验容量比例、坐标越界、空间重叠、Sub-Cube 互斥、依赖/并发约束、unplaced 和 mapping_rate。
- 映射评分加入 `transition_conflict_weight`，降低高频相邻/交替专家造成的切换与冲突风险。
- `scripts/generate_synthetic_model.py` 生成 DeepSeek/MoE 风格 metadata-only 大模型与 trace，用于证明大规模适配能力。

## 当前 main 关键增强

- `config/moe_config.py` 新增 shared/non-expert operator replication 开关和阈值：`enable_shared_operator_replication`、`shared_replication_min_volume`、`shared_replication_max_replicas`。
- `main.py` 新增 CLI：`--disable-shared-replication`、`--shared-replication-min-volume`、`--shared-replication-max-replicas`。
- `main.py` 新增 `--export-internal-ir`，生成轻量 `internal_ir.json`，用于答辩和交接说明。
- `src/model_parser.py` 的 trace adapter 支持 `traces`、`records`、`activations` 三类外层格式，以及 `active_experts`、`expert_ids`、`experts`、`topk_experts` 四类记录字段。
- `CubeConfig.inter_subcube_transfer_penalty` 已接入 simulator，用于表达跨 Sub-Cube 依赖传输惩罚。
- `optimized_mapping.json` 的 `metadata.shared_operator_replication` 记录启用状态、候选算子类型、请求复制的逻辑块数和实际额外物理副本数。
- 消融输出名已改为 `ablation_best_fit_replication_only_*` 和 `ablation_moe_without_replication_*`，分别表达“仅 Best Fit/复制”和“MoE 无复制”。

