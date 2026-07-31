# API 文档

## 1. `src/model_parser.py`

### `parse_model(model_path: Path, cube_config: CubeConfig) -> ParsedModel`
- 功能：解析JSON模型，构建DAG，切分Weight-Cube；ONNX 路径为轻量适配/元数据级读取，当前覆盖 initializer-backed `MatMul/Gemm` 与常见 arithmetic elementwise 等基础形态，不声明完整 ONNX 执行语义覆盖。
- 输出：`ParsedModel`，包含算子字典、拓扑序、依赖图、Weight-Cube集合。
- 元数据保留：解析器会保留 `bias_shape`、`inputs`、`outputs`、`onnx_op_type`、`initializer_name` 等字段，便于后续导出 internal IR、追溯 ONNX initializer，或适配真实模型来源。

### `parse_activation_trace(trace_path: Path) -> Dict[str, Any]`
- 功能：解析MoE激活轨迹，输出共现矩阵和频率统计。
- 兼容格式：顶层记录列表可使用 `traces`、`records`、`activations`；单条记录中的专家字段可使用 `active_experts`、`expert_ids`、`experts`、`topk_experts`。解析器会归一化这些字段以构建专家频率和共现统计。

---

## 2. `src/internal_ir.py`

### `build_internal_ir(parsed_model, activation_trace, cube_cfg, mapping=None, simulation=None) -> Dict[str, Any]`
- 功能：构建轻量级内部 IR，用于答辩、复核和调试时展示从模型解析到硬件映射、静态调度的完整链路。
- 输入：
  - `parsed_model`: `parse_model` 生成的 `ParsedModel`。
  - `activation_trace`: `parse_activation_trace` 生成的 trace 统计与记录。
  - `cube_cfg`: 当前 `CubeConfig`，用于记录 Sub-Cube 网格、容量、带宽和惩罚参数。
  - `mapping`: 可选 `MappingResult`，传入后写入 mapping rate、space utilization、parameter density、placement/unplaced 计数等摘要。
  - `simulation`: 可选 `SimulationResult`，传入后写入 latency、schedule task count、constraint validity 等摘要。
- 输出：可 JSON 序列化的 dict，通常由 CLI 导出为 `internal_ir.json`。该 IR 是项目级轻量中间表示，用于展示链路和关键摘要；完整 placements/schedule 仍以 `*_mapping.json`、`*_simulation.json` 和 `solution.json` 为准。它不是 MLIR Dialect 或 ISA。

---

## 3. `src/mapping_solver.py`

### `solve_mapping(parsed_model, activation_trace, cube_cfg, moe_cfg, packing_policy, enable_moe_optimization, enable_adaptive_replication) -> MappingResult`
- 功能：执行空间映射。
- 支持：
  - `packing_policy="first_fit"|"best_fit"`
  - `packing_policy="aspect_aware"`（通常由 Best Fit + 配置自动启用）
  - MoE互斥分组
  - 多起点互斥分组
  - 热子图精确分配（小规模精确搜索）
  - 动态 hot-subgraph top-k
  - swap/2-opt局部搜索
  - 模拟退火全局搜索
  - 并行多起点搜索
  - 冷热分层
  - 自适应复制（含副本预算约束）
  - shared/non-expert operator replication（大共享算子在额外副本预算内复制）
  - 分块量化与稀疏压缩（压缩比写入placement）
- 输出：`MappingResult`（placements、未映射列表、指标、元信息）。
- 关键元信息：`grouping_strategy`、`exact_hot_subgraph`、`hot_subgraph_topk`、`placement_dynamic_weights`、`packing_dynamic_weights`、`constraints`、`shared_operator_replication`。

`shared_operator_replication` 包含：

- `enabled`: shared/non-expert operator replication 是否在当前 mapping 中启用。
- `operator_types`: 可复制的共享算子类型，默认包含 `linear`。
- `min_volume`: 触发共享算子复制的最小逻辑体积。
- `max_replicas`: 单个符合条件共享算子的最大副本数。
- `requested_logical_cubes`: 请求复制的逻辑 Weight-Cube 数。
- `extra_physical_replicas`: 实际放置成功的额外物理副本数。
- `replicated_logical_cubes`: 最终拥有多个物理 placement 的非专家逻辑 Weight-Cube 数。

### `build_mutually_exclusive_groups(co_matrix: np.ndarray) -> List[List[int]]`
- 功能：构建专家互斥分组。

---

## 4. `src/simulator.py`

### `simulate(parsed_model, mapping, activation_trace, cube_cfg) -> SimulationResult`
- 功能：静态时序模拟。
- 约束：
  - 依赖屏障
  - Sub-Cube互斥
  - 切换惩罚
  - z层访问惩罚
  - 子立方体异构算力与带宽
- 扩展参数：
  - `overlap_transfer_compute`
  - `overlap_alpha`
  - `overlap_model_mode="linear"|"nonlinear_bandwidth_aware"`
  - `overlap_bw_power_law_alpha`
  - `overlap_z_depth_penalty`
  - `load_balance_weight`
  - `dispatch_policy="fifo"|"criticality"`
  - `criticality_weight`
  - `resource_pressure_weight`
- 输出指标：
  - `latency`
  - `space_utilization`
  - `parameter_density`（映射参数量 / 占用体积）
  - `temporal_utilization`
  - `pipeline_bubble_cycles`
  - `switching_penalty_cycles`
  - `inter_subcube_transfer_penalty_cycles`（跨 Sub-Cube 数据传输惩罚周期）
  - `memory_utilization`
  - `avg_bandwidth_utilization`
  - `max_bandwidth_utilization`
  - `effective_bandwidth_bytes_per_cycle`
  - `subcube_busy_imbalance`
  - `parallel_limit_wait_cycles`

---

## 5. `src/utils.py`

### 数据与统计
- `load_json` / `save_json`
- `build_cooccurrence_matrix`
- `compute_expert_frequency`

### 可视化
- `plot_cooccurrence_heatmap`
- `plot_mapping_slices`
- `plot_schedule_gantt`
- `plot_latency_distribution`
- `plot_subcube_contention`

---

## 6. `main.py`

### `run_pipeline(..., export_internal_ir=False) -> Dict[str, Any]`
- 功能：一键执行基线与优化策略，输出对比结果与图表。
- 关键输入：`model_path`、`trace_path`、`output_dir`、`cube_cfg_override`、`moe_cfg_override`、调度/overlap 参数、严格容量参数和 `export_internal_ir`。
- `export_internal_ir=True` 时，返回值的 `artifacts.internal_ir` 为 `internal_ir.json` 的绝对路径，`run_manifest.json` 中记录相对 artifact 名称。
- `run_pipeline` 支持覆盖 cube/moe 配置、overlap 模型、调度策略、resource pressure、严格容量与 seed。
- 每次新运行会输出 `baseline_solution.json`、`ablation_best_fit_replication_only_solution.json`、`ablation_moe_without_replication_solution.json`、`optimized_solution.json`、最终提交用 `solution.json` 和 `run_manifest.json`。

### CLI
```powershell
& $PY main.py --model data/sample_model.json --trace data/sample_trace.json --output outputs --profile
& $PY main.py --model data/sample_model.json --trace data/sample_trace.json --output outputs --search-trials 12 --parallel-workers 6
& $PY main.py --model data/sample_model.json --trace data/sample_trace.json --output outputs/smoke --profile --deterministic --search-trials 1 --parallel-workers 1 --local-restarts 1 --local-iters 5 --disable-sa --disable-parallel-search
& $PY main.py --model data/sample_model.json --trace data/sample_trace.json --output outputs/nonlinear --profile --overlap-transfer-compute --overlap-model-mode nonlinear_bandwidth_aware --resource-pressure-weight 0.2 --dispatch-policy criticality --criticality-weight 0.3
& $PY main.py --model data/sample_model.json --trace data/sample_trace.json --output outputs/shared_replication_smoke --profile --deterministic --cube-d 2 --strict-capacity --capacity-max-ratio 2.0 --replication-volume-budget-ratio 0.45 --shared-replication-min-volume 16777216 --shared-replication-max-replicas 2
& $PY main.py --model data/sample_model.json --trace data/sample_trace.json --output outputs/ir_smoke --profile --export-internal-ir
```

Internal IR CLI:

- `--export-internal-ir`: 在输出目录写入 `internal_ir.json`，用于展示 `ParsedModel`、activation trace、`CubeConfig`、`MappingResult` 和 `SimulationResult` 的串联关系。该文件适合答辩和 reviewer-facing 材料引用，但不代表 MLIR/ISA 级完整编译器输出。

Shared replication CLI:

- `--disable-shared-replication`: 关闭大共享/非专家算子复制。
- `--shared-replication-min-volume`: 设置触发复制的最小逻辑体积，默认 `16777216`。
- `--shared-replication-max-replicas`: 设置符合条件共享算子的最大副本数，默认 `2`。

---

## 7. 自动化脚本

### `scripts/tune_optuna.py`
- 功能：自动化单目标超参搜索，输出 `best_params.json` 与 `tuning_trials.csv`。
- 支持：多 trace 鲁棒目标、p95/p99 尾延时、holdout 验证、trace 权重、两阶段调参、`--n-jobs` 并行。
- 关键开关：`--disable-shared-replication` 会在调参和 holdout replay 中关闭 shared/non-expert operator replication，并写入 `best_params.json`。
- 环境护栏：入口会校验 `sys.executable` 是项目 `.venv` 的 Python，并固定 multiprocessing executable，避免误用系统 Python。

### `scripts/parallel_benchmark.py`
- 功能：并行批量评估多组配置，输出 `parallel_benchmark_summary.json` 与 `parallel_benchmark.csv`。

### `scripts/generate_materials.py`
- 功能：根据 `outputs/` 自动生成赛题材料：PPT提纲、2分钟讲稿、提交检查清单。

### `scripts/tune_optuna_multi.py`
- 功能：自动化多目标 Pareto 搜索，输出 `multiobjective_pareto.json`。
- 支持：多 trace 鲁棒目标、holdout top-k 评估、trace 权重、两阶段调参、`--n-jobs` 并行。
- 关键开关：`--disable-shared-replication` 会在 Pareto 参数和 holdout 参数中显式记录 `enable_shared_operator_replication=false`。
- 环境护栏：入口会校验 `sys.executable` 是项目 `.venv` 的 Python，并固定 multiprocessing executable，避免误用系统 Python。

### `scripts/short_term_optimize.py`
- 功能：低成本网格搜索，输出 `short_term_best.json`、`short_term_grid_results.csv`、`short_term_report.md`。

### `scripts/export_gate_traces.py`
- 功能：导出或模拟 MoE gate 激活轨迹。`--mode mock` 不依赖深度学习框架；`--mode torch` 可用自定义 factory 与 gate attribute path 采样真实模型。
