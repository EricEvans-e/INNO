# 技术实现与复现指南

## 1. 使用前提

本项目必须使用项目虚拟环境，不使用系统 Python：

```powershell
$PY = "E:/Users/Eric/Desktop/Inno/saidao2/.venv/Scripts/python.exe"
```

项目目录：

```text
E:/Users/Eric/Desktop/Inno/saidao2/cim_3d_scheduler
```

Windows PowerShell 下应从项目目录执行命令。若虚拟环境依赖未安装：

```powershell
& $PY -m pip install -r requirements.txt
```

## 2. 交付包结构

```text
没头绪_赛道二_赛题二_提交材料/
├── README_提交说明.md
├── source/
│   ├── main.py
│   ├── config/
│   ├── data/
│   ├── src/
│   ├── scripts/
│   ├── tests/
│   ├── requirements.txt
│   └── environment.yml
├── formal_result/
│   ├── solution.json
│   ├── optimized_mapping.json
│   ├── optimized_simulation.json
│   ├── comparison_metrics.json
│   ├── run_manifest.json
│   └── validation_report.json
├── paper/
│   ├── cim_3d_scheduler_design_report.pdf
│   └── latex_source/
├── docs/
│   ├── PROJECT_INTRODUCTION.md
│   ├── REPRODUCTION_AND_HANDOFF_GUIDE.md
│   └── PPT制作方案.md
└── evidence/
    ├── validation_summary.txt
    └── test_summary.txt
```

源码目录不包含 `.git`、`.venv`、`__pycache__`、`.pytest_cache`、编辑器配置和历史调参目录。正式结果只保留可审阅和可提交的 V5 primary 产物，避免把历史实验混入最终包。

## 3. 最小验证

### 3.1 单元测试

```powershell
& $PY -m pytest -q
```

当前基线为 `30 passed`，可能有若干第三方或字体相关 warning。失败时先确认当前目录、虚拟环境路径和依赖版本。

### 3.2 快速流水线

```powershell
& $PY main.py `
  --model data/sample_model.json `
  --trace data/sample_trace.json `
  --output outputs/guide_alignment_ir_smoke `
  --profile --deterministic `
  --search-trials 1 --parallel-workers 1 `
  --local-restarts 1 --local-iters 5 `
  --disable-sa --disable-parallel-search `
  --cube-d 2 --strict-capacity `
  --capacity-max-ratio 2.0 --export-internal-ir

& $PY scripts/validate_submission.py `
  --output outputs/guide_alignment_ir_smoke `
  --strict-capacity --capacity-max-ratio 2.0 `
  --report outputs/guide_alignment_ir_smoke/validation_report.json
```

成功后应看到 `validation_report.json` 中 `ok=true`、`errors=[]`，并生成 `internal_ir.json`。

## 4. 正式 V5 primary 复现

正式推荐结果目录为 `outputs/submission_v5_primary`。完整参数命令已保存在 `docs/SUBMISSION_PARAM_CARD_V5.md`，复现时必须注意当前 `main.py` 默认包含 post-V5 shared replication，因此 V5 历史结果的命令必须显式包含：

```text
--disable-shared-replication
```

推荐流程：

1. 打开 `docs/SUBMISSION_PARAM_CARD_V5.md`，复制 V5 primary 的完整 `main.py` 命令。
2. 确认模型为 `data/sample_model.json`，轨迹为 `outputs/trace_variants_mock/trace_balanced.json`。
3. 确认输出目录为新的临时目录，避免覆盖已保存的正式证据。
4. 运行主程序并等待 `solution.json`、`run_manifest.json` 和模拟结果生成。
5. 使用严格容量模式运行验证器。
6. 对比 `comparison_metrics.json`，确认指标和本交付包记录一致。

验证命令：

```powershell
& $PY scripts/validate_submission.py `
  --output outputs/submission_v5_primary `
  --strict-capacity --capacity-max-ratio 2.0 `
  --report outputs/submission_v5_primary/validation_report.json
```

## 5. 输出文件说明

| 文件 | 作用 |
| --- | --- |
| `solution.json` | 面向提交的映射与静态调度主文件 |
| `optimized_mapping.json` | 优化器产生的 Weight-Cube 空间映射、分组、副本和元数据 |
| `optimized_simulation.json` | 优化方案的调度、周期、冲突和利用率结果 |
| `comparison_metrics.json` | 基线与优化方案的指标对比 |
| `run_manifest.json` | 输入文件、参数、哈希和运行元数据 |
| `validation_report.json` | 独立提交验证结果 |
| `internal_ir.json` | 用于答辩和交接的轻量中间表示，可选生成 |

## 6. 官方激活轨迹接入

官方轨迹到达后，不要直接替换正式证据目录。先复制到临时目录，执行以下检查：

1. 确认 JSON 可解析。
2. 确认存在 `traces`、`records` 或 `activations` 顶层数组，或者明确记录数组位置。
3. 确认每条记录能解析出 Expert ID 列表。
4. 确认 `num_experts`、`top_k`、推理条数和模型配置一致。
5. 先运行快速流水线，再运行正式参数。
6. 通过严格验证器后，才更新论文和正式输出。

适配器目前兼容的字段包括：

```json
{
  "traces": [
    {"active_experts": [1, 7, 12, 20]}
  ]
}
```

也兼容 `expert_ids`、`experts` 和 `topk_experts` 等记录字段。若官方字段不同，应在 `src/model_parser.py` 增加适配，而不是在实验文档中手工改写数据。

## 7. 论文复现

论文源文件位于 `docs/paper/`，正式 PDF 为：

```text
docs/paper/output/cim_3d_scheduler_design_report.pdf
```

Overleaf 使用方式：

1. 上传整个 `docs/paper` 目录。
2. 编译器设置为 XeLaTeX。
3. 主文件设置为 `main.tex`。
4. 若使用本地编译，按 `docs/paper/README.md` 的 XeLaTeX/Biber/XeLaTeX 流程执行。

论文封面已填写：队名“没头绪”、成员“祝铭堃（南开大学）”、学校“南开大学”、指导教师“无”。

## 8. PPT 制作与展示

PPT 不建议重新发明数据，直接使用 `docs/paper/figures/figure1.png` 至 `figure5.png` 和 `comparison_metrics.json` 中的真实指标。逐页内容、讲稿和版式见：

```text
docs/PPT制作方案.md
```

建议演示顺序为：问题约束 -> 系统流程 -> 3D 资源模型 -> 映射算法 -> 静态调度 -> 验证器 -> 结果 -> 复现与边界。

## 9. 常见错误

### 使用了系统 Python

表现为依赖缺失、Windows 多进程行为不同或调参脚本拒绝启动。始终使用 `$PY` 变量调用。

### 把 V5.1 当作正式结果

V5.1 在自定义 `trace_balanced` 上延迟更低，但在官方轨迹和长轨迹验证前应标记为候选。正式提交包默认放 V5 primary。

### 忽略验证器

优化器能生成文件不代表文件符合提交约束。任何最终输出都必须保留 `validation_report.json` 并确认 `ok=true`、`errors=[]`。

### 把模拟数字写成芯片实测

`cycles` 是当前性能模型中的周期数。论文和 PPT 必须使用“模拟延迟”“模型估计”或“simulation cycles”等表述。

### 误删样例输入

样例模型和轨迹用于复现 smoke test，必须和源码一起保留。官方轨迹不应覆盖样例文件。
