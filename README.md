# cim_3d_scheduler

赛道二 / 赛题二：3D 异构 CIM 资源调度优化实现。输入 `model.json` 或 ONNX 与 MoE 激活轨迹，输出空间映射、静态调度、对比指标、图表、`solution.json` 和可复现 `run_manifest.json`。

截至 2026-04-29，最终提交建议以 [docs/SUBMISSION_PARAM_CARD_V5.md](docs/SUBMISSION_PARAM_CARD_V5.md) 与 `outputs/final_param_recommendation_v5.json` 为准。V5 主结果在 `outputs/submission_v5_primary/`，备选结果在 `outputs/submission_v5_backup/`，大模型适配证明在 `outputs/large_scale_v5/run_strict/`。

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
& $PY main.py --model data/sample_model.json --trace data/sample_trace.json --output outputs/v5_smoke_strict_venv --profile --deterministic --search-trials 1 --parallel-workers 1 --local-restarts 1 --local-iters 5 --disable-sa --disable-parallel-search --cube-d 2 --strict-capacity --capacity-max-ratio 2.0
& $PY scripts/validate_submission.py --output outputs/v5_smoke_strict_venv --strict-capacity --capacity-max-ratio 2.0 --report outputs/v5_smoke_strict_venv/validation_report.json
```

## V5 最终结果

主方案 `outputs/submission_v5_primary/`：

- optimized latency: 234,436
- latency improvement: 56.56%
- conflict_score: 2.0
- mapping_rate: 1.0
- capacity_ratio: 1.9862 <= 2.0
- validator: pass

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

```powershell
& $PY scripts/tune_optuna.py --model data/sample_model.json --traces outputs/trace_variants_mock/trace_base.json,outputs/trace_variants_mock/trace_hotspot.json,outputs/trace_variants_mock/trace_bursty.json --holdout-traces outputs/trace_variants_mock/trace_balanced.json --output outputs/tuning_v5_strict_single_venv --trials 32 --n-jobs 4 --enable-two-stage-tuning --auto-trace-weight --robust-worst-weight 0.2 --tail-p95-weight 0.08 --tail-p99-weight 0.12 --cube-d 2 --max-parallel-subcubes 9 --strict-capacity --capacity-max-ratio 2.0 --seed 2026
& $PY scripts/tune_optuna_multi.py --model data/sample_model.json --traces outputs/trace_variants_mock/trace_base.json,outputs/trace_variants_mock/trace_hotspot.json,outputs/trace_variants_mock/trace_bursty.json --holdout-traces outputs/trace_variants_mock/trace_balanced.json --holdout-topk 3 --output outputs/tuning_v5_strict_multi_venv --trials 16 --n-jobs 4 --enable-two-stage-tuning --auto-trace-weight --overlap-transfer-compute --robust-worst-weight 0.2 --tail-p95-weight 0.08 --tail-p99-weight 0.12 --cube-d 2 --max-parallel-subcubes 9 --strict-capacity --capacity-max-ratio 2.0 --seed 2026
```

完整 primary/backup/large-scale 复现命令见 [docs/SUBMISSION_PARAM_CARD_V5.md](docs/SUBMISSION_PARAM_CARD_V5.md)。

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

