# 设计报告：3D异构CIM资源调度优化（面向大模型推理）

## 1. 引言
赛题二关注在固定3D异构CIM资源空间中，将大模型计算流（尤其是MoE）进行静态映射与时序调度优化。难点不是单纯容量放置，而是“并行性-互斥性-切换开销-依赖屏障”的系统级耦合优化。

本实现面向赛题评价指标，强调：
- 完整部署链路可运行、可复现；
- MoE结构特性显式建模；
- 指标导向的可量化优化与消融验证。

## 2. 工作原理与数学建模

### 2.1 资源模型
设全局资源空间为：
- Sub-Cube数量：$N^2$
- 每个Sub-Cube维度：$D \times H \times W$
- 单个Sub-Cube在任意时刻仅能激活1个Weight-Cube

### 2.2 Weight-Cube切分模型
给定权重矩阵 $W \in \mathbb{R}^{m \times n}$，切分约束：
- 仅允许水平/垂直切分
- 子块尺寸满足 $h \le H, w \le W$

切分后得到集合 $\{C_k\}$，每个 $C_k=(h_k,w_k,d_k)$；本实现采用 $d_k=1$ 的分层放置。

### 2.3 优化目标
多目标优化（按赛题优先级）：
1. 最小化端到端时延 $L$
2. 最大化空间利用率 $U_s=\frac{\sum \text{placed volume}}{N^2DHW}$
3. 提升大模型适配性（映射率）

### 2.4 MoE共现建模
从激活轨迹构建共现矩阵 $M$：
- $M_{ij}$ 表示专家 $i,j$ 同时激活次数
- 当 $M_{ij}=0$ 时，视作可互斥同核候选

## 3. 结构设计

### 3.1 模块划分
- `model_parser.py`：模型与轨迹解析，依赖图构建，权重自动切分
- `mapping_solver.py`：映射算法（First Fit/Best Fit + MoE优化）
- `simulator.py`：静态时序模拟（依赖屏障/互斥/切换/气泡）
- `utils.py`：共现分析与可视化
- `main.py`：一键执行与基线对比

### 3.2 数据接口
- 输入：`model.json`, `activation_trace.json`
- 中间结果：`*_mapping.json`, `*_simulation.json`
- 输出：指标对比与可视化图

### 3.3 与通关指南工具链视角的对应关系

通关指南从完整 CIM 编译工具链角度讨论模型导入、中间表示、硬件映射、静态调度和工程验证。本项目的定位更窄：面向赛道二 / 赛题二，聚焦 MoE-style inference 的专家权重部署、3D Weight-Cube 映射和静态调度优化，而不是赛题一式 MLIR Dialect、ISA 生成或全栈编译器实现。

在本实现中，`ParsedModel` 承担模型导入后的结构化模型表示，包含算子、依赖 DAG、Weight-Cube 切分结果以及保留的 ONNX/JSON 元数据；轻量级 `src/internal_ir.py` 可通过 `build_internal_ir(parsed_model, activation_trace, cube_cfg, mapping, simulation)` 导出 `internal_ir.json`，用于串联模型、trace、硬件配置、映射结果与调度结果。映射阶段输出 `MappingResult` 和 `*_mapping.json`，调度仿真阶段输出 `SimulationResult` 和 `*_simulation.json`，最终 `solution.json` 作为提交/复核风格的空间映射与 schedule 汇总。

因此，本项目与通关指南的对应关系是“工具链关键环节的赛题二调度化实现”：模型导入有 `parse_model`，中间表示有轻量 internal IR，硬件映射有 `MappingResult`，静态调度有 `SimulationResult`，工程验证有 `scripts/validate_submission.py`、`run_manifest.json` 和指标/图表产物。更详细的交叉说明见 `docs/GUIDE_ALIGNMENT.md`。

## 4. 详细设计与实现

### 4.1 模型解析
- 支持JSON模型；ONNX 为轻量适配/元数据级读取，当前覆盖 initializer-backed `MatMul/Gemm`、常见 arithmetic elementwise 等基础形态，不声明完整 ONNX 执行语义覆盖
- 自动构建DAG依赖图（networkx）
- 对线性/专家权重做二维切分，生成Weight-Cube列表

#### 4.1.1 权重切分合规性说明

赛题约束"仅允许水平或垂直切分"。本实现的 `_split_matrix` 函数（`model_parser.py`）采用网格切分策略：对给定权重矩阵 $W \in \mathbb{R}^{m \times n}$，先按行方向以 `max_h` 为步长做水平切分，再对每个水平条带按列方向以 `max_w` 为步长做垂直切分，得到一系列轴对齐子矩形。

这一策略可解释为连续水平/垂直轴对齐切分形成的规则网格分块。每次切分操作本身是水平或垂直的，最终子块均为轴对齐矩形，不存在对角线或旋转切割。赛题约束旨在排除非轴对齐的任意切割；在该约束理解下，本实现的网格切分符合“仅使用水平或垂直切分”的要求。

### 4.2 编译映射
#### 基线算法
- First Fit：按顺序尝试首个可放置位置

#### 优化算法
- Best Fit：在候选位置中选择浪费最小项
- Aspect-aware Packing：在 Best Fit 基础上引入形状匹配与碎片惩罚，降低后续放置失败概率
- MoE共现优化：
  1. 共现矩阵统计
  2. 多起点互斥分组（共现为0优先同组，并兼顾负载均衡）
  3. 热子图精确分配 + 局部搜索/模拟退火得到组到Sub-Cube映射
- 冷热分层：热点专家优先放置在低z层（`z=0`）
- 自适应复制：按激活频率动态决定专家副本份数，并支持 shared/non-expert operator replication 在额外副本预算内复制大共享算子，平衡冲突、共享层热点与容量
- 动态放置偏置：综合共现冲突、Sub-Cube已用体积、分组一致性和副本分散度

### 4.3 配置验证模拟器
- 依赖屏障：下游任务等待上游所有Section完成
- Sub-Cube互斥：同一Sub-Cube同一时刻只执行1个Weight-Cube
- 切换代价：同核切换不同Weight-Cube引入固定惩罚
- 流水线气泡：记录资源等待导致的空转周期
- 调度策略：支持 FIFO 与关键路径优先，并可叠加 resource pressure
- 传输/计算重叠：支持线性 overlap 与带宽/z层感知的非线性 overlap 模型

### 4.4 可复现输出
每次新运行会输出：
- `solution.json`：推荐提交用的 optimized 空间映射与静态调度
- `*_solution.json`：baseline、ablation、optimized 各自的 solution 快照
- `run_manifest.json`：输入文件 hash、配置、运行环境与结果摘要

### 4.5 硬件执行单元抽象

本项目没有将通关指南中的 DMA、Elementwise、AnaArray、DigArray、FFT 建模为指令级执行单元，而是根据赛题二的 MoE 专家权重调度目标建立调度层抽象。

- DMA / 数据搬运：由 transfer cycles、带宽参数、transfer/compute overlap、`CubeConfig.inter_subcube_transfer_penalty` 和 `inter_subcube_transfer_penalty_cycles` 表达，重点刻画跨 Sub-Cube 依赖带来的数据移动代价。
- AnaArray / CIM array：由 `linear`、`moe_expert`、`matmul`、`gemm` 等 Weight-Cube 承载，表示主要矩阵权重在 CIM 阵列上的空间部署与执行占用。
- Elementwise / digital light tasks：由 `elementwise`、`router`、`merge` 等轻量任务表示，用于保留 MoE 路由、逐元素操作、专家输出合并等依赖关系和少量调度开销。
- DigArray：当前粒度下并入轻量数字任务或非专家算子的 compute cycles，不单独输出硬件指令。
- FFT：未建模，因为 FFT 不是 MoE expert weight deployment and scheduling 的核心算子，也不是当前评测链路的主要瓶颈。

blockwise quantization、sparsity 和 replication 在本报告中均作为容量/性能调度模型元数据使用：它们影响 Weight-Cube 体积、有效参数量、副本预算和延迟估计，但不构成完整硬件数值精度校准或真实芯片 ISA 级模拟的声明。

## 5. 优化论证与分析
- 互斥分组降低高共现专家同核冲突概率
- 热点专家低层部署减少访问附加代价
- 复制策略缓解热点资源争用，降低切换与排队延迟
- Best Fit在相同容量下提升有效装载率，降低碎片

### 5.1 算法复杂度分析

| 算法阶段 | 复杂度 | 说明 |
|---------|--------|------|
| ShelfPacker (2D bin-packing) | O(C × R) | C=cube数, R=shelf行数, R≪H |
| MoE互斥分组 | O(T × E² × G) | T=多起点试验数, E=专家数, G=分组数 |
| 热子图精确分配 | O(N^k) (branch-and-bound剪枝) | k=top_k, N=子立方体数, 实际远小于最坏情况 |
| 局部搜索 (swap/2-opt) | O(I × (G×N + G²)) | I=迭代次数, G=可移动分组数 |
| 模拟退火 | O(S × G²) | S=SA步数, 每步评估O(G²)冲突分数 |
| 调度器 | O(T × (P + log T)) | T=任务数, P=候选placement数 |

各阶段说明：
- **ShelfPacker**：对每个待放置cube，遍历shelf所有行寻找最优放置位置，行数R受子立方体高度和最小cube尺寸约束，实际远小于H。
- **MoE互斥分组**：多起点贪心，每轮对每个专家尝试放入已有分组或新建分组，需检查组内所有成员的共现关系。
- **热子图精确分配**：DFS枚举top-k个热点分组到子立方体的映射，branch-and-bound利用部分冲突下界剪枝，k=6、N=9时最坏情况9^6=531,441，实际剪枝后远小于此。
- **局部搜索**：每轮尝试所有单组迁移(G×N种)和所有组对交换(G²种)，迭代I轮收敛。
- **模拟退火**：每步随机扰动分组映射并评估冲突分数，S步终止。
- **调度器**：对每个任务评估所有候选placement(P个)，使用优先队列管理就绪任务。

## 6. 实验结果
详见 `EXPERIMENT.md`，含消融、指标表与可视化结果。

## 7. 总结
本项目实现了赛题要求的全流程可运行框架，并将MoE特性显式融入映射与调度。在示例场景中可稳定输出可追溯中间结果与量化指标，满足复现与答辩需求。

## 8. 参考文献
1. MoE架构相关论文（Switch Transformer / GShard / DeepSeek MoE公开资料）
2. 2D Bin Packing与静态调度经典方法
3. 赛题官方文档与约束说明
