# 提交检查清单（V5）

本文记录 2026-04-29 的 V5 提交状态。命令均使用项目虚拟环境：

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

## 结果摘要

- primary optimized latency: 234,436, improvement: 56.56%, capacity_ratio: 1.9862
- backup optimized latency: 258,401, improvement: 52.53%, capacity_ratio: 1.9862
- large-scale proof optimized latency: 64,863, improvement: 63.09%, capacity_ratio: 1.9427

## 复现入口

完整调参、复跑和 validator 命令见 `docs/SUBMISSION_PARAM_CARD_V5.md`。
