# Deep Optimization Update (2026-03-31)

## 1. 目标
在现有“副本 + 压缩 + 重叠”方案基础上，进一步提升项目能力：
- 映射层：增强 Weight-Cube 放置策略，降低潜在并发冲突并改善子块负载均衡。
- 调度层：加入关键路径优先调度能力，并提升调度器在大任务量下的可扩展性。
- 自动化层：将新增能力接入网格/Optuna 调参链路，支持一键探索。

## 2. 核心实现

### 2.1 映射器增强（src/mapping_solver.py）
新增“动态子块偏置”机制，在每次放置时综合以下信号：
- 冲突感知：候选 Sub-Cube 与已放置专家的共现冲突质量。
- 负载感知：候选 Sub-Cube 当前已用体积比例。
- 分组一致性：若开启 MoE 分组优化，惩罚跨组混放。
- 副本多样性：鼓励同一逻辑权重副本分散到不同 Sub-Cube。

新增配置权重（config/moe_config.py）：
- placement_conflict_weight
- placement_load_weight
- placement_group_penalty
- replica_diversity_penalty

### 2.2 调度器增强（src/simulator.py）
新增调度策略：
- fifo：保持原先行为。
- criticality：按 DAG 关键路径深度提升优先级。

新增参数：
- dispatch_policy: fifo | criticality
- criticality_weight: 关键路径优先权重

同时将 ready 队列从每轮排序改为堆结构（heapq），降低调度器复杂度开销。

### 2.3 主流程与调参脚本接入
- main.py
  - 新增 CLI：--dispatch-policy, --criticality-weight
- scripts/short_term_optimize.py
  - 新增网格维度：dispatch_policies, criticality_weights
- scripts/tune_optuna.py
  - 新增搜索参数：dispatch_policy, criticality_weight
  - 新增鲁棒目标：支持多 trace，且把 p95/p99 延时纳入目标函数
  - 新增搜索参数：placement_conflict_weight, placement_load_weight, placement_group_penalty, replica_diversity_penalty
- scripts/tune_optuna_multi.py
  - 新增搜索参数：dispatch_policy, criticality_weight
  - 新增鲁棒多目标：robust_latency/conflict/space（robust_latency 含 worst + p95 + p99）
  - 新增搜索参数：placement_conflict_weight, placement_load_weight, placement_group_penalty, replica_diversity_penalty

## 3. 验证结果

### 3.1 回归测试
命令：
```bash
python -m pytest -q
```
结果：
- 2026-04-29 验证：5 passed，1 个 Matplotlib `get_cmap` 弃用 warning

### 3.2 主流程冒烟（关键路径调度）
命令：
```bash
python main.py --model data/sample_model.json --trace data/sample_trace.json --output outputs/deep_main_smoke --dispatch-policy criticality --criticality-weight 0.4 --overlap-transfer-compute --overlap-alpha 0.8 --load-balance-weight 0.05
```
结果（optimized 对 baseline）：
- Latency improvement: 57.72%
- Space utilization improvement: 2.76%

### 3.3 网格调参（含新参数）
命令：
```bash
python scripts/short_term_optimize.py --model data/sample_model.json --trace outputs/trace_exporter_mock.json --output outputs/deep_opt_smoke --replica-ratios 0.15 --cold-quant-bits 4 --cold-sparsity 0.3 --overlap-flags true --overlap-alphas 0.7 --lb-weights 0.0 --dispatch-policies fifo,criticality --criticality-weights 0.0,0.3
```
结果：
- 4 trials 全部成功，最佳结果写入 outputs/deep_opt_smoke/short_term_best.json

### 3.4 多轨迹鲁棒网格（含新参数）
命令：
```bash
python scripts/short_term_optimize.py --model data/sample_model.json --traces outputs/trace_variants_mock/trace_base.json,outputs/trace_variants_mock/trace_hotspot.json,outputs/trace_variants_mock/trace_bursty.json --output outputs/deep_opt_robust_v2 --replica-ratios 0.15,0.25 --cold-quant-bits 4 --cold-sparsity 0.3 --overlap-flags true --overlap-alphas 0.7,1.0 --lb-weights 0.0 --dispatch-policies fifo,criticality --criticality-weights 0.0,0.3 --robust-worst-weight 0.2
```
结果：
- 16 trials 全部成功
- Best score: 602650.3981
- mean latency: 397601.0
- worst latency: 460198.0

### 3.5 鲁棒单目标 Optuna（含尾延时 + 动态排布权重）
命令：
```bash
python scripts/tune_optuna.py --model data/sample_model.json --traces outputs/trace_variants_mock/trace_base.json,outputs/trace_variants_mock/trace_hotspot.json,outputs/trace_variants_mock/trace_bursty.json --output outputs/tuning_robust_v2 --trials 12 --robust-worst-weight 0.2 --tail-p95-weight 0.08 --tail-p99-weight 0.12
```
结果：
- Best score: 594328.3123
- mean latency: 365269.67
- worst latency: 426150.0
- mean p95 latency: 174883.43
- mean p99 latency: 212951.89
- 关键新增搜索维度已生效：placement_conflict_weight / placement_load_weight / placement_group_penalty / replica_diversity_penalty

### 3.6 鲁棒多目标 Optuna（Pareto）
命令：
```bash
python scripts/tune_optuna_multi.py --model data/sample_model.json --traces outputs/trace_variants_mock/trace_base.json,outputs/trace_variants_mock/trace_hotspot.json,outputs/trace_variants_mock/trace_bursty.json --output outputs/tuning_multi_robust_v2 --trials 4 --overlap-transfer-compute --robust-worst-weight 0.2 --tail-p95-weight 0.08 --tail-p99-weight 0.12
```
结果：
- trials: 4
- pareto size: 1
- Pareto best robust latency: 711696.9603

### 3.7 第三轮强化：并行鲁棒调参与留出集验证
单目标鲁棒调参（20 trials，n_jobs=2，含 holdout）：
```bash
python scripts/tune_optuna.py --model data/sample_model.json --traces outputs/trace_variants_mock/trace_base.json,outputs/trace_variants_mock/trace_hotspot.json,outputs/trace_variants_mock/trace_bursty.json --holdout-traces outputs/trace_variants_mock/trace_balanced.json --output outputs/tuning_robust_v3 --trials 20 --n-jobs 2 --seed 2026 --robust-worst-weight 0.2 --tail-p95-weight 0.08 --tail-p99-weight 0.12
```
结果：
- best score: 538547.9401
- train mean latency: 329211.67
- train worst latency: 391538.0
- train mean p95/p99: 159969.78 / 193690.01
- holdout score: 349151.0980

留出集完整对比（应用最优参数跑全流程）：
```bash
python main.py --model data/sample_model.json --trace outputs/trace_variants_mock/trace_balanced.json --output outputs/run_best_v3_on_balanced --profile --overlap-transfer-compute --overlap-alpha 0.5661151654453002 --load-balance-weight 0.05220935338070645 --dispatch-policy criticality --criticality-weight 0.2707887892398476 --local-restarts 4 --local-iters 45 --search-trials 8 --hot-subgraph-topk 5 --placement-conflict-weight 1.0009099379265975 --placement-load-weight 0.6364746941126821 --placement-group-penalty 0.36795257819878924 --replica-diversity-penalty 0.5444728269541278
```
结果（balanced trace）：
- Baseline latency: 2,063,443
- Optimized latency: 455,866
- Latency improvement: 77.91%

多目标鲁棒调参（6 trials，n_jobs=2，含 holdout top-k 评估）：
```bash
python scripts/tune_optuna_multi.py --model data/sample_model.json --traces outputs/trace_variants_mock/trace_base.json,outputs/trace_variants_mock/trace_hotspot.json,outputs/trace_variants_mock/trace_bursty.json --holdout-traces outputs/trace_variants_mock/trace_balanced.json --holdout-topk 2 --output outputs/tuning_multi_robust_v3 --trials 6 --n-jobs 2 --seed 2026 --overlap-transfer-compute --robust-worst-weight 0.2 --tail-p95-weight 0.08 --tail-p99-weight 0.12
```
结果：
- trials: 6
- pareto size: 1
- pareto holdout 已自动写入 `multiobjective_pareto.json` 的 `holdout_eval`

### 3.8 第四轮强化：64轮鲁棒调参与提交参数卡
单目标鲁棒调参（64 trials，n_jobs=2）：
```bash
python scripts/tune_optuna.py --model data/sample_model.json --traces outputs/trace_variants_mock/trace_base.json,outputs/trace_variants_mock/trace_hotspot.json,outputs/trace_variants_mock/trace_bursty.json --holdout-traces outputs/trace_variants_mock/trace_balanced.json --output outputs/tuning_robust_v4 --trials 64 --n-jobs 2 --seed 2026 --robust-worst-weight 0.2 --tail-p95-weight 0.08 --tail-p99-weight 0.12
```
结果：
- best score: 528320.7890
- train mean latency: 323839.0
- holdout score: 331889.3414

多目标鲁棒调参（16 trials，n_jobs=2）：
```bash
python scripts/tune_optuna_multi.py --model data/sample_model.json --traces outputs/trace_variants_mock/trace_base.json,outputs/trace_variants_mock/trace_hotspot.json,outputs/trace_variants_mock/trace_bursty.json --holdout-traces outputs/trace_variants_mock/trace_balanced.json --holdout-topk 3 --output outputs/tuning_multi_robust_v4 --trials 16 --n-jobs 2 --seed 2026 --overlap-transfer-compute --robust-worst-weight 0.2 --tail-p95-weight 0.08 --tail-p99-weight 0.12
```
结果：
- trials: 16
- pareto size: 1

主流程完整复现（主/备）后指标：
- 主方案（full params）在 balanced 上 latency: 223,387，较 baseline 提升 79.83%
- 备选方案（full params）在 balanced 上 latency: 282,574，较 baseline 提升 79.47%

产物：
- 提交参数卡：`docs/SUBMISSION_PARAM_CARD_V4.md`
- 结构化推荐：`outputs/final_param_recommendation_v4.json`

## 4. 新能力总结
- 支持“映射冲突/负载联合优化 + 调度关键路径优先”的端到端能力。
- 支持在网格与 Optuna 中自动搜索新调度参数。
- 支持多 trace 鲁棒目标，并把尾延时（p95/p99）纳入调优目标。
- 支持自动搜索动态排布权重（冲突/负载/分组/副本多样性）。
- 支持并行调参（`n_jobs`）、可复现实验（`seed`）与自动剪枝（MedianPruner）。
- 支持留出集自动验证（single best + pareto top-k）。
- 支持主流程完整参数复现（含压缩/副本预算/SA参数/动态排布权重）。
- 调度器复杂度更优（heap ready queue），更适合扩大 trace 规模。
- 2026-04-23 后代码已继续补强：`main.py` 支持 `solution.json`/`run_manifest.json` 导出、确定性模式、非线性带宽感知 overlap、resource pressure、复制压力阈值和动态 hot-subgraph 配置；Optuna 脚本支持两阶段调参与 trace 权重。

## 5. 下一步建议（可直接执行）
1. 用真实大规模 trace（>=50k inferences）跑 `short_term_optimize.py` 与 `tune_optuna_multi.py`，验证堆调度在大样本下的收益。
2. 在真实 trace 上提升 `tune_optuna.py`/`tune_optuna_multi.py` 试验数到 50~200，以稳定收敛 Pareto 前沿。
3. 增加 holdout trace（如 balanced）做泛化验证，筛选对尾延时更稳健的参数集。
