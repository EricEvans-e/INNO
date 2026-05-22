# 设计报告：3D异构CIM资源调度优化（面向大模型推理）

## 1. 引言
赛题二关注在固定3D异构CIM资源空间中，将大模型计算流（尤其是MoE）进行静态映射与时序调度优化。难点不是单纯容量放置，而是“并行性-互斥性-切换开销-依赖屏障”的系统级耦合优化。

本实现面向冲一等奖目标，强调：
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

## 4. 详细设计与实现

### 4.1 模型解析
- 支持JSON模型（并兼容ONNX读取）
- 自动构建DAG依赖图（networkx）
- 对线性/专家权重做二维切分，生成Weight-Cube列表

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

## 5. 优化论证与分析
- 互斥分组降低高共现专家同核冲突概率
- 热点专家低层部署减少访问附加代价
- 复制策略缓解热点资源争用，降低切换与排队延迟
- Best Fit在相同容量下提升有效装载率，降低碎片

## 6. 实验结果
详见 `EXPERIMENT.md`，含消融、指标表与可视化结果。

## 7. 总结
本项目实现了赛题要求的全流程可运行框架，并将MoE特性显式融入映射与调度。在示例场景中可稳定输出可追溯中间结果与量化指标，满足复现与答辩需求。

## 8. 参考文献
1. MoE架构相关论文（Switch Transformer / GShard / DeepSeek MoE公开资料）
2. 2D Bin Packing与静态调度经典方法
3. 赛题官方文档与约束说明
