# Submission Parameter Card V4 (Round-4 Optimized)

## 1. 目标
本参数卡用于比赛提交阶段的可复现实验配置，给出：
- 主方案（单目标鲁棒 64 轮最优）
- 备选方案（多目标 Pareto 候选）
- 风险边界与切换规则

数据来源：
- outputs/tuning_robust_v4/*
- outputs/tuning_multi_robust_v4/*
- outputs/run_best_v4_fullparams_on_balanced/*
- outputs/run_backup_v4_fullparams_on_balanced/*

## 2. 主方案（Primary）
来源：`outputs/tuning_robust_v4/best_params.json`

### 2.1 参数
- overlap_alpha: 0.6312970155665848
- load_balance_weight: 0.06869248060875036
- dispatch_policy: fifo
- criticality_weight: 0.6988552999439486
- local_search_restarts: 8
- local_search_max_iters: 46
- parallel_trials: 11
- hot_subgraph_top_k: 3
- replication_volume_budget_ratio: 0.3963516192285268
- enable_simulated_annealing: true
- sa_steps: 135
- sa_init_temp: 1.3642435642842063
- sa_cooling: 0.929683295949587
- hot_quant_bits: 8
- cold_quant_bits: 4
- hot_sparsity_ratio: 0.23525578002080225
- cold_sparsity_ratio: 0.44193673437299547
- placement_conflict_weight: 1.1655753536429578
- placement_load_weight: 0.42351662898599407
- placement_group_penalty: 0.667959906476027
- replica_diversity_penalty: 0.4384820373759949

### 2.2 训练集鲁棒指标（3 traces）
- best score: 528320.7889936191
- mean latency: 323839.0
- worst latency: 381284.0
- mean p95 latency: 156240.42
- mean p99 latency: 189823.34

### 2.3 留出集（balanced）完整复跑结果
来源：`outputs/run_best_v4_fullparams_on_balanced/comparison_metrics.json`
- Baseline latency: 1,107,598
- Optimized latency: 223,387
- Latency improvement: 79.83%
- Optimized conflict_score: 2.0
- Optimized space_utilization: 0.1710

## 3. 备选方案（Backup）
来源：`outputs/tuning_multi_robust_v4/multiobjective_pareto.json` 的 Pareto Trial 6

### 3.1 参数
- overlap_alpha: 0.8975297876877559
- load_balance_weight: 0.07954472159355522
- dispatch_policy: fifo
- criticality_weight: 0.09978678020995542
- local_search_restarts: 6
- local_search_max_iters: 22
- parallel_trials: 4
- hot_subgraph_top_k: 7
- replication_volume_budget_ratio: 0.12617960742392262
- enable_simulated_annealing: true
- sa_steps: 86
- sa_init_temp: 0.7405834302661823
- sa_cooling: 0.9224690255495351
- hot_quant_bits: 8
- cold_quant_bits: 5
- hot_sparsity_ratio: 0.18481427760381133
- cold_sparsity_ratio: 0.4384558219399425
- placement_conflict_weight: 1.1503786998465892
- placement_load_weight: 0.001975709562107908
- placement_group_penalty: 0.1738197456370962
- replica_diversity_penalty: 0.16565410667178604

### 3.2 留出集（balanced）完整复跑结果
来源：`outputs/run_backup_v4_fullparams_on_balanced/comparison_metrics.json`
- Baseline latency: 1,376,559
- Optimized latency: 282,574
- Latency improvement: 79.47%
- Optimized conflict_score: 5.0
- Optimized space_utilization: 0.1398

## 4. 主备切换规则（Risk Boundaries）
默认使用主方案；满足任一条件时切换备选：
1. 需要更低空间利用率目标（期望 `space_utilization <= 0.15`）。
2. SA 搜索预算受限（希望更短搜索时长，降低 local/parallel 搜索规模）。

默认保持主方案；若出现以下风险信号需回退或重调：
1. `optimized.conflict_score > 4`
2. `optimized.pipeline_bubble_ratio > 0.996`
3. `optimized.avg_bandwidth_utilization < 0.45`

## 5. 一键复现命令
### 5.1 主方案
```bash
python main.py --model data/sample_model.json --trace outputs/trace_variants_mock/trace_balanced.json --output outputs/run_best_v4_fullparams_on_balanced --profile --overlap-transfer-compute --overlap-alpha 0.6312970155665848 --load-balance-weight 0.06869248060875036 --dispatch-policy fifo --criticality-weight 0.6988552999439486 --local-restarts 8 --local-iters 46 --search-trials 11 --hot-subgraph-topk 3 --replication-volume-budget-ratio 0.3963516192285268 --sa-steps 135 --sa-init-temp 1.3642435642842063 --sa-cooling 0.929683295949587 --hot-quant-bits 8 --cold-quant-bits 4 --hot-sparsity-ratio 0.23525578002080225 --cold-sparsity-ratio 0.44193673437299547 --placement-conflict-weight 1.1655753536429578 --placement-load-weight 0.42351662898599407 --placement-group-penalty 0.667959906476027 --replica-diversity-penalty 0.4384820373759949
```

### 5.2 备选方案
```bash
python main.py --model data/sample_model.json --trace outputs/trace_variants_mock/trace_balanced.json --output outputs/run_backup_v4_fullparams_on_balanced --profile --overlap-transfer-compute --overlap-alpha 0.8975297876877559 --load-balance-weight 0.07954472159355522 --dispatch-policy fifo --criticality-weight 0.09978678020995542 --local-restarts 6 --local-iters 22 --search-trials 4 --hot-subgraph-topk 7 --replication-volume-budget-ratio 0.12617960742392262 --sa-steps 86 --sa-init-temp 0.7405834302661823 --sa-cooling 0.9224690255495351 --hot-quant-bits 8 --cold-quant-bits 5 --hot-sparsity-ratio 0.18481427760381133 --cold-sparsity-ratio 0.4384558219399425 --placement-conflict-weight 1.1503786998465892 --placement-load-weight 0.001975709562107908 --placement-group-penalty 0.1738197456370962 --replica-diversity-penalty 0.16565410667178604
```

## 6. 结论
截至 2026-04-29，推荐提交配置为主方案（Round-4 primary）。其在留出集上实现了更优 latency、更低 conflict，且提升幅度稳定；备选方案作为低空间利用率偏好或搜索预算受限场景的替代。
