# 路演PPT提纲（自动生成）

> 本文件由 `scripts/generate_materials.py --output outputs --docs docs` 生成。截至 2026-04-29，本文数值来自根目录 `outputs/comparison_metrics.json` 的小样例烟测。最终 V4 提交参数与留出集指标见 `SUBMISSION_PARAM_CARD_V4.md`。

## 1. 赛题与目标
- 赛道二：3D异构CIM资源调度优化
- 目标：在保证映射率的同时降低时延并提升利用率

## 2. 方法总览
- 热子图精确分配（近似ILP子问题）
- swap/2-opt局部搜索 + 并行多起点
- 模拟退火全局搜索
- 自适应副本预算控制
- 分块量化 + 稀疏压缩
- 异构算力/带宽时序仿真

## 3. 核心结果
- baseline latency: 34117.00
- optimized latency: 9732.00
- latency improvement: 71.47%
- temporal utilization improvement: 245.83%

## 4. 可解释分析
- expert调用频次、延迟分布、sub-cube争用、带宽利用率
- 冲突分数对比与消融

## 5. 提交价值
- 可复现脚本完整
- 参数可调、可扩展到更大模型
- 输出材料完备（图表 + JSON + 报告）
