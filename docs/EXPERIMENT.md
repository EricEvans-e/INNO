# 实验报告（含消融）

## 1. 实验设置
- 模型：`data/sample_model.json`（含共享层 + 16个MoE专家）
- 轨迹：`data/sample_trace.json`（Top-K激活）
- 资源配置：`N=3, D=8, H=W=4096`
- 比较方案：
  1. 基线：First Fit，无MoE优化，无复制
  2. 优化：Best Fit + 热子图精确分配 + swap/2-opt + 模拟退火 + 并行多起点 + 复制自适应 + 量化稀疏压缩

## 2. 指标定义
- 端到端时延：`latency`
- 空间利用率：`space_utilization`
- 时间利用率：`temporal_utilization`
- 切换开销：`switching_penalty_cycles`
- 流水线气泡：`pipeline_bubble_cycles`
- 内存利用率：`memory_utilization`
- 带宽利用率：`avg_bandwidth_utilization`

## 3. 消融实验设计

### A. 无MoE优化
- 关闭互斥分组与冷热分层，仅保留基础装箱策略。

### B. 无复制策略
- 固定每个Weight-Cube仅1份，不做热点复制。

### C. 完整优化（推荐提交版本）
- 开启所有优化开关。

## 4. 结果记录方式
运行以下命令：
```bash
python main.py --model data/sample_model.json --trace data/sample_trace.json --output outputs --profile
```
读取 `outputs/comparison_metrics.json` 与可视化图完成表格。

## 5. 小样例烟测结果

以下表格来自根目录 `outputs/comparison_metrics.json`，用于说明算法链路与消融格式，不等同于最终 V4 提交参数的留出集复跑结果。

| 方案 | 时延(cycles) | 空间利用率 | 时间利用率 | 切换代价(cycles) | 气泡(cycles) | 冲突分数 |
|---|---:|---:|---:|---:|---:|---:|
| 基线（First Fit, 无MoE, 无复制） | 34117 | 0.1259 | 0.1142 | 447 | 1350314 | 93 |
| 消融A（Best Fit, 无MoE, 有复制） | 9732 | 0.1293 | 0.3949 | 315 | 180113 | 24 |
| 消融B（Best Fit, 有MoE, 无复制） | 10115 | 0.1259 | 0.3799 | 264 | 162957 | 39 |
| 完整优化（Best Fit, 有MoE, 有复制） | 9732 | 0.1293 | 0.3949 | 315 | 180113 | 24 |

相对基线改进（完整优化）：
- 时延降低：71.47%
- 空间利用率提升：2.76%
- 时间利用率提升：245.83%
- 切换代价降低：29.53%
- 流水线气泡降低：86.66%

## 6. V4提交参数留出集结果

最终提交建议见 `SUBMISSION_PARAM_CARD_V4.md` 与 `outputs/final_param_recommendation_v4.json`。

| 方案 | 数据来源 | Baseline latency | Optimized latency | 时延降低 | 冲突分数 | 空间利用率 |
|---|---|---:|---:|---:|---:|---:|
| 主方案 Primary | `outputs/run_best_v4_fullparams_on_balanced/comparison_metrics.json` | 1,107,598 | 223,387 | 79.83% | 2.0 | 0.1710 |
| 备选方案 Backup | `outputs/run_backup_v4_fullparams_on_balanced/comparison_metrics.json` | 1,376,559 | 282,574 | 79.47% | 5.0 | 0.1398 |

## 7. 可视化结果
- `outputs/cooccurrence_heatmap.png`
- `outputs/optimized_mapping_slices.png`
- `outputs/optimized_schedule_gantt.png`
- `outputs/optimized_latency_distribution.png`
- `outputs/optimized_subcube_contention.png`

## 8. 结论
完整优化策略通常在以下方面优于基线：
1. 时延降低（冲突减少 + 热点并发能力提升）
2. 空间装载更均衡（Best Fit降低碎片）
3. 切换与气泡可控（互斥分组避免高冲突同核）

补充说明：根 `outputs/comparison_metrics.json` 的 toy 规模样例下，“完整优化”和“消融A”在时延上接近，说明热子图精确分配与并行局部搜索已能捕获多数收益；复制策略收益更依赖更大规模并发负载。建议在真实大模型与长轨迹上继续用 `scripts/tune_optuna.py` 做阈值与预算联合搜索。
