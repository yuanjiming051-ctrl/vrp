# VRPTW DQN-ALNS 框架概览

本文档汇总仓库中的主要模块，并梳理求解车路时间窗问题（VRPTW）时的算法顺序，便于快速理解整体架构。

## 顶层执行流程
- `main.py` 是程序入口：首先加载 `VRPTWInstance`，然后构造 `DQNALNSSolver` 并触发求解，求解结束后再用 `VRPTWDecoder` 解码最优序列及绘图。【F:main.py†L1-L83】

## 数据读取与解码层
- `VRPTWInstance` 负责解析 Solomon 格式的数据文件，拆分车辆信息与客户信息，并构建包含仓库与客户的距离矩阵，是所有下游模块的基础。【F:vrptw_instance.py†L4-L93】
- `VRPTWDecoder` 基于 NumPy 与 Numba 提供快速解码能力，可把任意客户访问序列映射为车辆数、总距离、路径结构等指标，同时支持从 Deep 空间序列恢复到 Full 空间序列。【F:decoder.py†L1-L200】

## Deep 构造与双空间表示
- `DeepConstructor` 通过距离递归划分、2-opt 微调等操作，把原始客户集合压缩为若干“深度子路线”，并生成 Deep 空间实例；`map_deep_to_full` 等工具负责在两个空间之间映射，支持后续的协同演化。【F:deep_constructor.py†L1-L121】【F:dqn_alns_solver.py†L194-L209】

## DQN + ALNS 协同主循环
- `DQNALNSSolver.solve` 是核心：
  1. 利用多启发式 `InitialSolutionGenerator` 初始化 Full 空间种群，并随机生成 Deep 空间种群，分别评估成本作为起点。【F:dqn_alns_solver.py†L303-L318】
  2. 在每一轮（最长 1000 轮或超时结束）中，先基于成本选出少量精英个体作为后续操作的主体。【F:dqn_alns_solver.py†L324-L361】
  3. 对 Deep 精英执行遗传算子与改进的破坏-修复（ALNS）多次尝试与局部搜索，逐步降低 Deep 空间成本，并在成功时回写种群。【F:dqn_alns_solver.py†L364-L432】
  4. 将更新后的 Deep 解映射回 Full 空间，若能严格改进则替换对应个体，确保两个空间共享信息。【F:dqn_alns_solver.py†L433-L470】
  5. 在 Full 空间对精英个体调用 DQN 评估的破坏-修复动作、最差路径优化、局部搜索等策略；若产生改进则更新种群与经验回放，用于后续 DQN 学习。【F:dqn_alns_solver.py†L471-L620】
  6. 使用当前最佳 Full 解重新定义 Deep 空间聚类，强制 Deep 种群与最新结构保持一致，从而形成“Full→Deep→Full”的循环反馈。【F:dqn_alns_solver.py†L621-L693】
  7. 在循环过程中维护 10/30 分钟快照与历史最优成本轨迹，最终返回最优 Full 序列及统计信息供外部展示。【F:dqn_alns_solver.py†L295-L340】【F:dqn_alns_solver.py†L694-L753】

## 动作生成与局部改进模块
- `AlnsOperators` 提供多种破坏与修复算子（随机、Shaw、整路径移除等）以及局部搜索策略，既用于 Deep 空间短跑也用于 Full 空间 DQN 评估的动作集合。【F:dqn_alns_solver.py†L68-L90】【F:dqn_alns_solver.py†L552-L610】
- `DQNAgent` + `SimpleNodeAggregator` 组合构成动作价值网络：前者管理 replay buffer 与参数更新，后者对节点特征做聚合，最终预测每个破坏-修复动作的 Q 值，指导 Full 空间的选择次序。【F:dqn_alns_solver.py†L92-L153】
- `WorstRouteOptimizer`、遗传算法参数与多样性控制逻辑负责在 Full 空间中进一步强化最差路径、控制种群多样性，防止陷入局部最优。【F:dqn_alns_solver.py†L154-L199】【F:dqn_alns_solver.py†L520-L590】

通过上述分层结构，系统实现了“Deep 空间快速探索 + Full 空间精细评估 + DQN 动态调度”的协同搜索流程。
