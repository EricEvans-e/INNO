# 通关指南对齐说明

本文面向答辩、评审和后续交接，说明本项目与 `saidao2/参考/交大赛道2赛题2通关指南.md` 的对应关系。结论先行：本项目聚焦赛道二 / 赛题二的 3D 异构 CIM 资源调度，核心对象是 MoE expert 权重部署、Weight-Cube 映射和静态调度优化；它不是赛题一意义上的完整 MLIR/ISA 编译器，也不宣称已经覆盖端到端硬件指令生成。

## 1. 定位边界

通关指南从“编译工具链”的大视角出发，包含模型导入、中间表示、图优化、硬件映射、代码生成和硬件执行单元适配等内容。其中 MLIR、Dialect、ISA 生成等内容更接近赛题一的完整编译器架构设计。

本项目的边界如下：

- 目标问题：赛题二的 3D heterogeneous CIM resource scheduling，重点是 MoE-style inference 中专家权重的空间部署和时序调度。
- 输入输出：从 JSON/ONNX 风格模型描述与 activation trace 进入，输出 mapping、simulation、`solution.json`、`run_manifest.json` 以及可选 `internal_ir.json`。
- 中间表示：使用轻量级 internal IR 解释 ParsedModel、MappingResult、SimulationResult 的链路，不构建 MLIR Dialect，也不生成真实 ISA。
- 评价重点：容量约束、空间利用、参数密度、Sub-Cube 互斥、依赖调度、跨 Sub-Cube 传输惩罚、MoE 热点与副本策略。
- 正式推荐：当前正式提交推荐仍是 `outputs/submission_v5_primary`；post-V5 shared/non-expert operator replication 仅作为后续探索，除非在 balanced/holdout 上取得稳定收益，否则不提升为正式方案。

## 2. 通关指南要点与本项目实现对照

| 通关指南要点 | 本项目对应实现 / 产物 | 对齐说明 |
| --- | --- | --- |
| 模型导入 | `src/model_parser.py`、`parse_model`、`ParsedModel` | 支持 JSON，并兼容 ONNX 风格算子元数据；保留 `bias_shape`、`inputs`、`outputs`、`onnx_op_type`、`initializer_name` 等字段，便于后续替换真实模型来源。 |
| 激活轨迹导入 | `parse_activation_trace`、sample trace、后续官方 trace adapter | 兼容顶层 `traces`、`records`、`activations`；单条记录兼容 `active_experts`、`expert_ids`、`experts`、`topk_experts`，使官方 2026 年 7 月 trace 到达后可低成本接入。 |
| 中间表示 | `src/internal_ir.py`、`build_internal_ir(...)`、`internal_ir.json` | 采用轻量级、答辩友好的 JSON IR，串联模型、trace、硬件配置、映射和调度结果；定位是可解释中间产物，不是 MLIR Dialect。 |
| 硬件映射 | `src/mapping_solver.py`、`MappingResult`、`*_mapping.json` | 将 Weight-Cube 映射到 3D Sub-Cube 空间，包含 best-fit/aspect-aware packing、MoE 互斥分组、热点专家低层部署、自适应副本和容量约束。 |
| 静态调度 | `src/simulator.py`、`SimulationResult`、`*_simulation.json`、`solution.json` | 在依赖屏障、Sub-Cube 互斥、切换惩罚、z 层访问惩罚、带宽与 transfer/compute overlap 模型下生成可验证静态 schedule。 |
| 自动化调优 | `scripts/tune_optuna.py`、`scripts/tune_optuna_multi.py`、`outputs/final_param_recommendation_v5.json` | 使用单目标 / 多目标搜索优化调度参数，并保留 V5 replay 参数卡，支持复现实验和答辩解释。 |
| 工程验证 | `scripts/validate_submission.py`、`run_manifest.json`、`comparison_metrics.json`、可视化图表 | 校验容量、映射边界、重叠、Sub-Cube 互斥、依赖关系、unplaced cubes 和 mapping_rate；每次运行保留输入 hash 与关键参数。 |

## 3. 硬件执行单元抽象

通关指南提到 DMA、Elementwise、AnaArray、DigArray、FFT 等执行单元。本项目没有按真实硬件 ISA 建模每个单元，而是围绕赛题二调度目标建立等价抽象：

- DMA / 数据搬运：通过 transfer cycles、带宽模型、`inter_subcube_transfer_penalty` 和 `inter_subcube_transfer_penalty_cycles` 表达跨 Sub-Cube 数据移动成本。
- AnaArray / CIM array：通过 `linear`、`moe_expert`、`matmul`、`gemm` 等 Weight-Cube 任务建模主要矩阵计算负载。
- DigArray / 数字阵列：在当前粒度中并入非专家算子和调度任务的 compute cycles，不单独生成指令级数字阵列操作。
- Elementwise / digital light tasks：通过 `elementwise`、`router`、`merge` 等轻量任务表达 MoE 路由、融合和逐元素操作对依赖链的影响。
- FFT：未建模。原因是 FFT 不是 MoE expert weight deployment and scheduling 的核心瓶颈，也不是当前赛题二样例链路的重点算子。

这种抽象服务于“资源部署 + 静态调度 + 约束校验”，而非宣称已经完成真实芯片上的指令级执行模拟。

## 4. 精度与性能边界

项目中出现的 blockwise quantization、sparsity、replication 等字段主要是调度容量与性能模型的元数据：

- blockwise quantization：用于估算 Weight-Cube 体积、压缩比和容量占用，不代表已经完成硬件数值误差标定。
- sparsity：用于描述有效参数量、压缩收益和调度负载变化，不代表完整稀疏算子硬件实现。
- replication：用于缓解热点专家或共享算子的资源争用，优化排队、切换和并行度，不等价于指南中以数值精度补偿为目标的权重复制校准方案。

因此，本文档中的“精度/性能对齐”应理解为调度层面的容量与延迟建模边界，而不是完整硬件数值精度承诺。正式报告中如引用这些能力，应使用“模型元数据”“容量估计”“性能调度模型”等措辞。

## 5. 正式推荐状态

当前正式推荐保持不变：

- 推荐提交目录：`outputs/submission_v5_primary`
- 参数说明：`docs/SUBMISSION_PARAM_CARD_V5.md`
- 结构化推荐：`outputs/final_param_recommendation_v5.json`

post-V5 shared/non-expert operator replication 已进入 `main`，并在 smoke 场景有通过验证的产物；但在 `trace_balanced` 上尚未超过 V5 primary。因此，答辩或提交材料中应把它表述为“后续优化探索”，而不是当前正式提交方案。
