# 提交检查清单（V5）

本文记录 V5 提交状态及后续对齐验证。命令均使用项目虚拟环境：

```powershell
$PY = "E:/Users/Eric/Desktop/Inno/saidao2/.venv/Scripts/python.exe"
```

## 代码与环境

- [x] 使用 `saidao2/.venv`，不依赖系统 Python
- [x] `requirements.txt` 依赖已在 `.venv` 中可用
- [x] `main.py` 支持严格容量、Cube 参数化、V5 调参复现参数
- [x] `solution.json` 顶层包含 `weight_cubes` 与 `schedule`
- [x] `scripts/validate_submission.py` 可独立验证提交目录

## V5 验证

- [x] `& $PY -m pytest -q`: 9 passed
- [x] `outputs/v5_smoke_strict_venv`: validator pass
- [x] `outputs/submission_v5_primary`: validator pass
- [x] `outputs/submission_v5_backup`: validator pass
- [x] `outputs/large_scale_v5/run_strict`: validator pass

## 2026-05-22 main 后续验证

- [x] `& $PY -m pytest -q`: 14 passed, 18 warnings
- [x] `outputs/v6_shared_replication_smoke`: validator pass, capacity_ratio 1.9862 <= 2.0
- [x] shared/non-expert operator replication 已接入 `main.py` CLI 与 `optimized_mapping.json` 元信息
- [x] `outputs/v6_shared_replication_balanced_candidate`: validator pass, latency 234,436，未超过 V5 primary
- [x] `outputs/v6_shared_replication_balanced_budget045`: validator pass, latency 333,003，放宽共享复制预算后退化

## 2026-05-27 赛题澄清与增强

以下为 2026-05-27 阶段性复核记录，当前最终回归仍以本文后续“通关指南对齐检查”和最终交付前验证命令为准。

- [x] `& $PY -m pytest -q`: 15 passed, 18 warnings
- [x] `CubeConfig.inter_subcube_transfer_penalty` 已建模（默认 0，不影响现有行为）
- [x] `parameter_density` 指标已添加（映射参数量 / 占用体积）
- [x] 设计报告新增 4.1.1 权重切分合规性说明
- [x] 设计报告新增 5.1 算法复杂度分析

## 通关指南对齐检查

- [x] `internal_ir.json` 可通过 `--export-internal-ir` 生成，并包含 `operators`、`weight_cubes`、`hardware`、`activation_trace`。
- [x] `GUIDE_ALIGNMENT.md` 说明本项目聚焦赛题二的 3D 异构 CIM 资源调度，而不是赛题一的 MLIR/ISA 编译链。
- [x] `DESIGN_REPORT.md` 描述轻量内部 IR、硬件执行单元抽象、精度/性能边界。
- [x] `parse_activation_trace` 支持 `traces`、`records`、`activations` 三类 trace 外层格式。
- [x] 正式推荐仍为 V5 primary；post-V5 smoke 只作为后续优化验证，不替代官方结果。

验证命令：

```powershell
& $PY -m pytest -q
& $PY main.py --model data/sample_model.json --trace data/sample_trace.json --output outputs/guide_alignment_ir_smoke --profile --deterministic --search-trials 1 --parallel-workers 1 --local-restarts 1 --local-iters 5 --disable-sa --disable-parallel-search --cube-d 2 --strict-capacity --capacity-max-ratio 2.0 --export-internal-ir
& $PY scripts/validate_submission.py --output outputs/guide_alignment_ir_smoke --strict-capacity --capacity-max-ratio 2.0 --report outputs/guide_alignment_ir_smoke/validation_report.json
```

## 2026-06-08 收口验证

- [x] `& $PY -m pytest -q`: 20 passed, 19 warnings
- [x] `outputs/submission_v5_primary`: validator pass，`ok=true`，`errors=[]`，`capacity_ratio=1.986206896551724`
- [x] `outputs/guide_alignment_ir_smoke`: `--export-internal-ir` smoke pass，baseline latency 16,874，optimized latency 9,619，latency improvement 43.00%
- [x] `outputs/guide_alignment_ir_smoke`: validator pass，`ok=true`，`errors=[]`，`capacity_ratio=1.986206896551724`
- [x] `outputs/guide_alignment_ir_smoke/internal_ir.json`: schema `cim_3d_scheduler.ir.v1`，22 operators，22 weight cubes，并已写入 `run_manifest.json`

## 2026-06-09 拿奖导向增强候选

- [x] `outputs/submission_v5_1_candidate`: optimized latency 231,672，较 V5 primary 的 234,436 降低 2,764 cycles（约 1.18%）
- [x] `outputs/submission_v5_1_candidate`: validator pass，`ok=true`，`errors=[]`，`capacity_ratio=1.986206896551724`
- [x] V5.1 核心变化：`--replication-volume-budget-ratio 0.35` + `--disable-shared-replication`
- [x] `scripts/validate_submission.py` 加固 schedule-to-placement 一致性、重复 physical id、重复/缺失 task id、零长度任务检查
- [x] `tests/test_v5_compliance.py`: validator 负例覆盖新增检查
- [x] `scripts/tune_optuna.py` / `scripts/tune_optuna_multi.py`: 新增 `--disable-shared-replication`，调参和 holdout 参数会显式记录该开关
- [x] `tests/test_tuning_scripts.py`: 覆盖 tuning 脚本 shared replication 关闭、参数落盘、venv Python 护栏
- [x] `outputs/tuning_v5_1_balanced_targeted_8_single`: 小规模单进程 sanity search 完成，最佳 holdout balanced latency 373,249，未超过 V5.1，不晋级
- [x] Windows venv 诊断：`sys.executable` 为项目 `.venv`，`sys._base_executable` 可显示系统 Python；调参脚本已固定 `multiprocessing.set_executable(sys.executable)` 并在误用系统 Python 时 fail fast
- [x] 最终回归：`& $PY -m pytest -q` 为 30 passed, 24 warnings；`py_compile` 覆盖主入口、配置、parser、IR、mapping、simulator、tuning、validator、synthetic model 脚本
- [x] 最终 validator：`outputs/submission_v5_primary`、`outputs/submission_v5_1_candidate`、`outputs/guide_alignment_ir_smoke` 均 `ok=true`、`errors=[]`、`warnings=[]`

## 最终产物

- [x] `outputs/submission_v5_primary/solution.json`
- [x] `outputs/submission_v5_primary/run_manifest.json`
- [x] `outputs/submission_v5_primary/validation_report.json`
- [x] `outputs/submission_v5_backup/solution.json`
- [x] `outputs/submission_v5_backup/run_manifest.json`
- [x] `outputs/submission_v5_backup/validation_report.json`
- [x] `outputs/large_scale_v5/synthetic_model.json`
- [x] `outputs/large_scale_v5/synthetic_trace.json`
- [x] `outputs/large_scale_v5/run_strict/solution.json`
- [x] `outputs/final_param_recommendation_v5.json`
- [x] `docs/SUBMISSION_PARAM_CARD_V5.md`
- [x] `docs/SUBMISSION_PARAM_CARD_V5_1.md`
- [x] `docs/GUIDE_ALIGNMENT.md`
- [x] `outputs/guide_alignment_ir_smoke/internal_ir.json`
- [x] `outputs/guide_alignment_ir_smoke/validation_report.json`

## 结果摘要

- primary optimized latency: 234,436, improvement: 56.56%, capacity_ratio: 1.9862
- V5.1 candidate optimized latency: 231,672, improvement: 57.08%, capacity_ratio: 1.9862
- balanced-targeted 8-trial sanity search best holdout latency: 373,249（负结果，不作为提交候选）
- backup optimized latency: 258,401, improvement: 52.53%, capacity_ratio: 1.9862
- large-scale proof optimized latency: 64,863, improvement: 63.09%, capacity_ratio: 1.9427
- guide-alignment smoke optimized latency: 9,619, improvement: 43.00%, capacity_ratio: 1.9862

## 复现入口

完整 V5 调参、复跑和 validator 命令见 `docs/SUBMISSION_PARAM_CARD_V5.md`；V5.1 增强候选命令见 `docs/SUBMISSION_PARAM_CARD_V5_1.md`。
