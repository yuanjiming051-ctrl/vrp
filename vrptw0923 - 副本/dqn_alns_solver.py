# File: dqn_alns_solver.py

import random
import numpy as np
import torch
import torch.nn as nn

from vrptw_instance import VRPTWInstance
from initial_solution import InitialSolutionGenerator
from decoder import VRPTWDecoder
from deep_constructor import DeepConstructor, DeepVRPTWInstance, map_deep_to_full, adaptive_route_reduction, visualize_reduction_comparison
from alns_ops import AlnsOperators
from dqn_agent import DQNAgent
from state_encoder import SimpleNodeAggregator
from worst_route_optimizer import WorstRouteOptimizer
from cluster_visualizer import ClusterVisualizer
# from local_search import apply_local_search

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNALNSSolver:
    def __init__(self,
                 instance: VRPTWInstance,
                 pop_size=50,  # 扩充默认种群数量从原来的小规模到50
                 delta: int = 2,
                 node_feat_dim: int = 5,
                 hidden_dim: int = 128,
                 embed_dim: int = 128,
                 deep_iters: int = 5,
                 deep_search_multiplier: float = 2.0,  # 降维空间搜索次数倍数
                 iters=20,
                 full_iters: int = 50,
                 eps_start: float = 1.0,
                 eps_end: float = 0.1,
                 eps_decay: float = 0.98,
                 destroy_ops=None,
                 repair_ops=None,
                 enable_local_search: bool = True,
                 local_search_type: str = 'adaptive',
                 local_search_frequency: int = 5,
                 verbose: bool = True,
                 distance_threshold: float = 5.0,
                 # 多样化策略参数
                 enable_restart: bool = True,
                 restart_threshold: int = 100,  # 无改进轮数阈值
                 diversity_threshold: float = 0.1,  # 种群多样性阈值
                 elite_ratio: float = 0.2,  # 精英保留比例
                 # 遗传算法参数
                 ga_crossover_rate: float = 0.8,  # 交叉概率
                 ga_mutation_rate: float = 0.1,   # 变异概率
                 ga_tournament_size: int = 3,     # 锦标赛选择大小
                 ga_elite_count: int = 2):        # 精英个体数量（每次只选择最好的2个进行动作）
        self.instance = instance
        self.pop_size = pop_size
        self.delta = delta
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.deep_iters = deep_iters
        self.deep_search_multiplier = deep_search_multiplier  # 降维空间搜索次数倍数
        self.iters = iters
        self.full_iters = full_iters
        self.distance_threshold = float(distance_threshold)

        self.alns_ops = AlnsOperators(instance, random)

        self.destroy_ops_full = [
            lambda chrom: self.alns_ops.random_removal(chrom, max(5, len(chrom) // 12)),
            # lambda chrom: self.alns_ops.worst_removal(chrom, max(5, len(chrom) // 15), sample_frac=0.30),
            lambda chrom: self.alns_ops.shaw_removal(chrom, max(5, len(chrom) // 15))
        ]

        # 初始化 destroy_ops_deep，使用相同的操作符但参数适配深度空间（优化：增加移除数量）
        self.destroy_ops_deep = [
            lambda chrom: self.alns_ops.random_removal(chrom, max(2, len(chrom) // 8)),  # 从1/20增加到1/8
            lambda chrom: self.alns_ops.shaw_removal(chrom, max(2, len(chrom) // 6)),    # 从1/15增加到1/6
            lambda chrom: self.alns_ops.complete_route_removal(chrom, target_routes=1)   # 使用完整路径移除操作
        ]

        # 修复操作需要考虑是否在Deep空间执行
        self.repair_ops = [
            # 当传入inst参数时，使用专门的Deep空间修复逻辑
            lambda destroyed, removed, inst: self._repair_in_space(destroyed, removed, inst, 
                                                                  lambda d, r: self.alns_ops.greedy_repair(d, r)),
            lambda destroyed, removed, inst: self._repair_in_space(destroyed, removed, inst, 
                                                                  lambda d, r: self.alns_ops.regret_insertion(d, r, k=2))
        ]

        self.action_size = len(self.destroy_ops_full) * len(self.repair_ops)

        # -------------------------------
        # 3) 构建 Q 网络 = GAT + Linear
        #    - GAT: SimpleNodeAggregator 将 (N, node_feat_dim) -> (embed_dim,)
        #    - Linear: nn.Linear(embed_dim, action_size) -> Q 值
        # -------------------------------
        # 3.1) DQNAgent 只用来存储 replay buffer、learn、soft_update，自己不再使用 agent_full.act()
        self.verbose = verbose  # 在使用前先定义，避免 AttributeError
        self.agent_full = DQNAgent(
            state_size=self.embed_dim + 3,
            action_size=self.action_size,
            seed=0,
            lr=1e-4,  # 降低学习率以获得更稳定的训练
            gamma=0.995,  # 增加折扣因子，更重视长期奖励
            tau=1e-3,
            update_every=4,
            verbose=self.verbose)

        # 3.2) 第一句：GAT 编码器
        self.aggregator = SimpleNodeAggregator(node_feat_dim=self.node_feat_dim,
                                               hidden_dim=self.hidden_dim,
                                               embed_dim=self.embed_dim).to(device)

        # 4) ε-greedy 参数
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        # 3.3) 第二句：线性层，映射 embed_dim -> action_size+    #    Linear head: embed_dim -> action_size
        self.q_linear = nn.Linear(self.embed_dim + 3, self.action_size).to(device)

        # 只把 q_linear 注入 agent，optimizer 也只管理它
        self.agent_full.qnetwork_local = self.q_linear
        self.agent_full.qnetwork_target = nn.Linear(self.embed_dim + 3, self.action_size).to(device)
        self.agent_full.optimizer = torch.optim.Adam(
            self.q_linear.parameters(), lr=1e-4)  # 与DQNAgent保持一致的学习率
        # 3.4) 把这两部分组合为 qnetwork_local、qnetwork_target
        #      qnetwork_local(state) 先跑 GAT 再跑 Linear
        #      self.agent_full.qnetwork_local = nn.Sequential(
        #         self.aggregator,
        #          self.q_linear
        #      ).to(device)
        # target 网络：同样结构，但要新建一份 GAT+Linear
        #     self.agent_full.qnetwork_target = nn.Sequential(
        #        SimpleNodeAggregator(node_feat_dim=self.node_feat_dim,
        #                             hidden_dim=self.hidden_dim,
        #                             embed_dim=self.embed_dim).to(device),
        #         nn.Linear(self.embed_dim, self.action_size).to(device)
        #      ).to(device)

        # 3.5) 创建 optimizer，优化 qnetwork_local 的所有参数
        #     self.agent_full.optimizer = torch.optim.Adam(
        #         self.agent_full.qnetwork_local.parameters(), lr=5e-4
        #     )

        # -------------------------------
        # 4) ε-greedy 参数
        # -------------------------------
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        # -------------------------------
        # 5) 局部搜索参数
        # -------------------------------
        # 局部搜索参数（优化后）
        self.enable_local_search = enable_local_search
        self.local_search_type = local_search_type
        self.local_search_frequency = max(1, local_search_frequency)  # 最少1轮执行一次，最大化局部搜索频率
        # self.verbose = verbose  # 已在上文初始化，避免重复设置
        
        # 多样化策略参数
        self.enable_restart = enable_restart
        self.restart_threshold = restart_threshold
        self.diversity_threshold = diversity_threshold
        self.elite_ratio = elite_ratio
        self.no_improvement_count = 0  # 无改进计数器
        self.last_best_cost = float('inf')  # 上次最佳成本
        
        # 遗传算法参数
        self.ga_crossover_rate = ga_crossover_rate
        self.ga_mutation_rate = ga_mutation_rate
        self.ga_tournament_size = ga_tournament_size
        self.ga_elite_count = ga_elite_count

        # -------------------------------
        # 6) 解码器 & 全局随机生成器
        # -------------------------------
        self.decoder = VRPTWDecoder(self.instance)
        self.init_gen = InitialSolutionGenerator(self.instance)
        
        # -------------------------------
        # 6.1) 最差路径优化器
        # -------------------------------
        self.worst_route_optimizer = WorstRouteOptimizer(self.instance, self.decoder)
        
        # -------------------------------
        # 6.2) 节点聚合可视化器
        # -------------------------------
        self.cluster_visualizer = ClusterVisualizer(output_dir="cluster_visualization")

        # -------------------------------
        # 7) 构建 Deep 空间实例
        # -------------------------------
        self.dc = DeepConstructor(self.instance, delta=self.delta)
        deepdata = self.dc.run()
        self.cluster_label = self.dc.VC_new
        self.Nd = len(deepdata['customer'])
        self.deep_inst = DeepVRPTWInstance(deepdata)
        self.init_gen_deep = InitialSolutionGenerator(self.deep_inst)

        # 7.1) 构造 cluster_label 数组（原客户数 -> 深度节点索引）
        n_customers = len(self.instance.ordinary_customers)
        self.cluster_label = [0] * len(self.instance.ordinary_customers)
        for deep_idx, cluster in enumerate(self.dc.VC_new):
            for cust in cluster:
                self.cluster_label[cust - 1] = deep_idx

    def extract_node_features(self, chrom, space='full'):
        """
        对 Full 空间或 Deep 空间的染色体，提取每个节点的特征：
        特征向量 = [xcoord, ycoord, demand, ready_time, due_date]，
        返回 Tensor 形状 (N, node_feat_dim)。chrom 仅用于决定路线顺序，但特征提取不依赖 chrom 顺序本身。
        """
        if space == 'deep':
            nodes = self.deep_inst.ordinary_customers
        else:
            nodes = self.instance.ordinary_customers

        N = len(nodes)
        feat = torch.zeros((N, self.node_feat_dim), device=device)
        for i in range(N):
            c = nodes[i]
            feat[i, 0] = c['xcoord']
            feat[i, 1] = c['ycoord']
            feat[i, 2] = c['demand']
            feat[i, 3] = c['ready_time']
            feat[i, 4] = c['due_date']
        return feat  # Tensor (N, node_feat_dim)

    def _evaluate_full(self, seq_full):
        """
        在 Full 空间上计算一个 Full 序列的总距离成本。
        """
        # 规范化序列，确保为 Python int，避免 numpy.int64 等类型在 JIT 中出现问题

        seq_full = [int(x) for x in seq_full]

        result = self.decoder.decode_solution(seq_full, strategy='fast')
        return result['total_distance']

    def _repair_in_space(self, destroyed, removed, inst, repair_func):
        """
        根据不同空间选择合适的修复策略
        
        Args:
            destroyed: 被破坏后的序列
            removed: 被移除的节点列表
            inst: 问题实例，用于判断是否为Deep空间
            repair_func: 修复函数
            
        Returns:
            修复后的序列
        """
        # 判断是否在Deep空间操作
        if inst is self.deep_inst:
            # 在Deep空间中，使用传入的repair_func直接修复
            # Deep空间中的节点索引是连续的，可以直接使用
            return repair_func(destroyed, removed)
        else:
            # 在Full空间中，使用传入的repair_func直接修复
            return repair_func(destroyed, removed)
            
    def _evaluate_deep(self, seq_deep):
        """
        直接在Deep空间解码计算成本，使用decoder的space和ds参数。
        使用当前最新的Deep空间定义。
        """
        # 优先使用动态更新的deep_data，否则回退到初始的
        deep_data = {
            'VC_new': getattr(self, 'current_VC_new', self.dc.VC_new)
        }
        
        # 直接使用decoder的space和ds参数进行解码
        seq_deep = [int(x) for x in seq_deep]  # 确保序列中的元素是Python int类型
        result = self.decoder.decode_solution(seq_deep, strategy='fast', space='deep', ds=deep_data)
        return result['total_distance']

    def solve(self):
        """
        主流程（种群协同进化）：
          1) 初始化 pop_full（Full 解种群）与 pop_deep（Deep 解种群）
          2) 对 deep_iters 轮：
             A.1) 随机选 i: Deep 空间短跑 ALNS → 更新 pop_deep[i]
             A.2) 遍历所有 pop_deep[k]：Deep→Full → 更新 pop_full[k]
             B.1) 随机选 j: Full 空间短跑 DQN-LNS → 更新 pop_full[j]
             B.2) 遍历所有 pop_full[k]: Full→Deep → 更新 pop_deep 中最差个体
          3) 返回最优 Full 解 & 成本 & 历史最优 Full 成本轨迹
        """
        # ===============================
        # 1) 初始化种群 pop_full & pop_deep
        # ===============================

        start_time = time.time()
        ten_minutes = 10 * 6000
        thirty_minutes = 30 * 6000

        # 记录 10 分钟和 30 分钟时的数据
        ten_minute_data = None
        thirty_minute_data = None

        pop_full = []
        pop_full_cost = []
        for _ in range(self.pop_size):
            sol_full = self.init_gen.generate_random_solution()
            cost_full = self._evaluate_full(sol_full)
            pop_full.append(sol_full)
            pop_full_cost.append(cost_full)

        pop_deep = []
        pop_deep_cost = []
        for _ in range(self.pop_size):
            sol_deep = random.sample(range(self.Nd), self.Nd)
            cost_deep = self._evaluate_deep(sol_deep)
            pop_deep.append(sol_deep)
            pop_deep_cost.append(cost_deep)

        best_full_history = []

        # ===============================
        # 2) 种群协同进化循环（共 deep_iters 轮）
        # ===============================
        for round_idx in range(1, 1000 + 1):
            elapsed_time = time.time() - start_time

            # 检查是否超过 30 分钟
            if elapsed_time > thirty_minutes:
                print("已达到 30 分钟，停止迭代。")
                break

            # 记录 10 分钟时的数据
            if elapsed_time > ten_minutes and ten_minute_data is None:
                best_full_idx = int(np.argmin(pop_full_cost))
                ten_minute_data = {
                    'best_sequence': pop_full[best_full_idx],
                    'best_cost': pop_full_cost[best_full_idx],
                    'cost_history': best_full_history.copy()
                }

            # ===============================
            # 注意：GA操作已移至各自的处理阶段中执行
            # ===============================
            if self.verbose:
                print(f"\n[Round {round_idx}] 开始精英选择和优化操作...")

            # ===============================
            # 精英选择：只选择最好的两个个体进行后续ALNS/DQN动作
            # ===============================
            # 获取Deep空间的精英个体（最好的2个）
            elite_deep_individuals, elite_deep_costs, elite_deep_indices = self.get_elite_individuals(
                pop_deep, pop_deep_cost, count=self.ga_elite_count)
            
            # 获取Full空间的精英个体（最好的2个）
            elite_full_individuals, elite_full_costs, elite_full_indices = self.get_elite_individuals(
                pop_full, pop_full_cost, count=self.ga_elite_count)
            
            if self.verbose:
                print(f"  选择精英个体进行动作:")
                print(f"    Deep精英: 个体{elite_deep_indices}, 成本{[f'{c:.2f}' for c in elite_deep_costs]}")
                print(f"    Full精英: 个体{elite_full_indices}, 成本{[f'{c:.2f}' for c in elite_full_costs]}")

            # ---------------------------------------
            # A.1) 在 Deep 空间对精英个体进行短跑 ALNS → 更新 pop_deep[精英索引]
            # ---------------------------------------
            individuals_to_optimize = elite_deep_indices
            num_to_optimize = len(individuals_to_optimize)
            
            if self.verbose:
                print(f"  [Deep ALNS] 选择所有 {num_to_optimize} 个个体进行优化")
            
            # ---------------------------------------
            # A.0) Deep空间GA操作：对选中的个体执行遗传算法
            # ---------------------------------------
            if self.verbose:
                print(f"  [Deep GA] 对Deep空间种群执行GA操作...")
            
            # 执行Deep空间GA操作
            pop_deep, pop_deep_cost = self.ga_evolve_selected_individuals(
                pop_deep, pop_deep_cost, space='deep'
            )
            
            if self.verbose:
                print(f"  [Deep GA] GA操作完成，种群已更新")
            
            # 对每个选中的个体进行Deep空间ALNS优化
            for opt_idx, i in enumerate(individuals_to_optimize):
                curr_deep = pop_deep[i].copy()
                curr_deep_cost = self._evaluate_deep(curr_deep)
                best_deep = curr_deep.copy()
                best_deep_cost = curr_deep_cost
                
                if self.verbose:
                    print(f"    [个体 {i}] 初始Deep序列: {curr_deep}, 初始成本: {curr_deep_cost:.2f}")

                # 早停机制已移除
                
                for iter_idx in range(self.deep_iters):
                    # 增强搜索：每次迭代进行多次操作尝试
                    iter_best_deep = curr_deep.copy()
                    iter_best_cost = curr_deep_cost
                    
                    # 尝试多个不同的破坏-修复组合
                    for attempt in range(3):  # 每次迭代尝试3次不同操作
                        # 随机选择破坏操作
                        op_d_idx = random.randrange(len(self.destroy_ops_deep))
                        destroy_op_name = self.destroy_ops_deep[op_d_idx].__name__
                        destroyed, removed = self.destroy_ops_deep[op_d_idx](curr_deep)
                        
                        # 随机选择修复操作
                        op_r_idx = random.randrange(len(self.repair_ops))
                        repair_op_name = self.repair_ops[op_r_idx].__name__
                        new_deep = self.repair_ops[op_r_idx](destroyed, removed, self.deep_inst)
                        new_deep_cost = self._evaluate_deep(new_deep)
                        
                        # 如果找到更好的解，更新当前迭代最佳
                        if new_deep_cost < iter_best_cost:
                            iter_best_deep = new_deep.copy()
                            iter_best_cost = new_deep_cost
                            if self.verbose:
                                print(f"      Deep迭代 {iter_idx+1}-{attempt+1}: 使用 {repair_op_name}, 移除 {removed}, 新成本: {new_deep_cost:.2f} (改进!)")
                        elif self.verbose:
                            print(f"      Deep迭代 {iter_idx+1}-{attempt+1}: 使用 {repair_op_name}, 移除 {removed}, 新成本: {new_deep_cost:.2f}")
                    
                    # 对当前迭代最佳解进行增强的局部搜索
                    if len(iter_best_deep) > 3:  # 只对足够长的序列进行局部搜索
                        # 1) 2-opt局部搜索（反转子序列）
                        for local_i in range(len(iter_best_deep)):
                            for local_j in range(local_i+2, min(local_i+6, len(iter_best_deep))):  # 限制搜索范围提高效率
                                local_deep = iter_best_deep.copy()
                                local_deep[local_i:local_j+1] = local_deep[local_i:local_j+1][::-1]
                                local_cost = self._evaluate_deep(local_deep)
                                if local_cost < iter_best_cost:
                                    iter_best_deep = local_deep.copy()
                                    iter_best_cost = local_cost
                        
                        # 2) 简单的relocate操作（移动单个节点）
                        for move_idx in range(len(iter_best_deep)):
                            node = iter_best_deep[move_idx]
                            for insert_idx in range(len(iter_best_deep)):
                                if insert_idx != move_idx and abs(insert_idx - move_idx) > 1:
                                    local_deep = iter_best_deep.copy()
                                    local_deep.pop(move_idx)
                                    local_deep.insert(insert_idx if insert_idx < move_idx else insert_idx-1, node)
                                    local_cost = self._evaluate_deep(local_deep)
                                    if local_cost < iter_best_cost:
                                        iter_best_deep = local_deep.copy()
                                        iter_best_cost = local_cost
                                    if self.verbose:
                                        pass
                                        #print(f"        -> 局部搜索改进: 成本 {local_cost:.2f}")
                                    break
                            if iter_best_cost < curr_deep_cost:  # 找到改进就跳出
                                break
                    
                    # 更新全局最佳和当前解
                    if iter_best_cost < best_deep_cost:
                        best_deep = iter_best_deep.copy()
                        best_deep_cost = iter_best_cost
                        if self.verbose:
                            print(f"        -> 迭代 {iter_idx+1} 找到新的最优解! 最佳成本更新为: {best_deep_cost:.2f}")
                    
                    curr_deep = iter_best_deep.copy()
                    curr_deep_cost = iter_best_cost
                    
                    # 早停机制已移除

                # 用 best_deep 更新 pop_deep[i]
                original_cost = pop_deep_cost[i]
                pop_deep[i] = best_deep.copy()
                pop_deep_cost[i] = best_deep_cost
                
                if self.verbose:
                    improvement = original_cost - best_deep_cost
                    print(f"    [个体 {i}] 优化完成: {original_cost:.2f} -> {best_deep_cost:.2f} (改进: {improvement:.2f})")
            
            if self.verbose:
                print(f"  [Deep ALNS] 本轮优化完成，当前Deep种群成本: {[f'{cost:.2f}' for cost in pop_deep_cost]}")
            # ---------------------------------------
            # A.2) Deep→Full：遍历所有 pop_deep[k]，若映射后更优则更新 pop_full[k]
            # ---------------------------------------
            # 获取当前 Deep 空间的映射关系
            vc_map = getattr(self, 'current_VC_new', self.dc.VC_new)
            if self.verbose:
                print(f"  [Deep→Full映射] 开始将Deep种群映射到Full空间...")
            
            deep_to_full_updates = 0
            for k in range(self.pop_size):
                full_from_deep = map_deep_to_full(pop_deep[k], vc_map)
                cost_full_from_deep = self._evaluate_full(full_from_deep)
                
                if self.verbose:
                    print(f"    个体 {k}: Deep序列 {pop_deep[k]} -> Full序列长度 {len(full_from_deep)}, 成本 {cost_full_from_deep:.2f}")

                # 多样性保护：除了第一轮外，避免产生重复的最小值
                updated = False
                if round_idx == 1:
                    # 第一轮允许任何改进
                    if cost_full_from_deep < pop_full_cost[k]:
                        pop_full[k] = full_from_deep
                        pop_full_cost[k] = cost_full_from_deep
                        updated = True
                else:
                    # 后续轮次：只有严格小于当前种群最小值时才允许更新
                    current_min_cost = min(pop_full_cost)
                    if cost_full_from_deep < pop_full_cost[k] and cost_full_from_deep < current_min_cost:
                        pop_full[k] = full_from_deep
                        pop_full_cost[k] = cost_full_from_deep
                        updated = True
                
                if updated:
                    deep_to_full_updates += 1
                    if self.verbose:
                        print(f"      -> 更新Full个体 {k}! 新成本: {cost_full_from_deep:.2f}")
            
            if self.verbose:
                print(f"  [Deep→Full映射] 完成，共更新了 {deep_to_full_updates} 个Full个体")

            # ---------------------------------------
            # B.0) Full空间GA操作：对Full空间种群执行遗传算法
            # ---------------------------------------
            if self.verbose:
                print(f"  [Full GA] 对Full空间种群执行GA操作...")
            
            # 执行Full空间GA操作
            pop_full, pop_full_cost = self.ga_evolve_selected_individuals(
                pop_full, pop_full_cost, space='full'
            )
            
            if self.verbose:
                print(f"  [Full GA] GA操作完成，种群已更新")
            
            # 应用局部搜索操作到Full空间种群
            if self.verbose:
                print(f"  [Full LS] 对Full空间种群应用局部搜索操作...")
            
            pop_full, pop_full_cost = self.apply_local_search_operations(
                pop_full, pop_full_cost, max_iterations=10
            )
            
            if self.verbose:
                print(f"  [Full LS] 局部搜索操作完成")
            
            # ---------------------------------------
            # B.1) 在 Full 空间对精英个体进行短跑 DQN-LNS → 更新 pop_full[精英索引]
            # ---------------------------------------
            if self.verbose:
                print(f"  [Full DQN] 对精英个体进行DQN-LNS操作...")
            
            # 对每个精英个体进行多次DQN迭代
            elite_iterations_per_individual = self.full_iters // self.ga_elite_count
            for elite_idx_pos, j in enumerate(elite_full_indices):
                if self.verbose:
                    print(f"    处理精英个体 {j} (第{elite_idx_pos+1}/{self.ga_elite_count}个精英)")
                
                for full_iter_idx in range(elite_iterations_per_individual):
                    curr_full = pop_full[j].copy()
                    curr_full_cost = pop_full_cost[j]
                    # 在解码前进行类型规范化，避免 numpy.int64 等类型潜在问题
                    try:
                        curr_full = [int(x) for x in curr_full]
                    except Exception:
                        pass
                    best_full_local = curr_full.copy()
                    best_full_local_cost = curr_full_cost

                    # 1) 编码当前状态
                    feat_f = self.extract_node_features(curr_full, space='full')
                    emb = self.aggregator(feat_f)

                    res = self.decoder.decode_solution(curr_full, strategy='fast')
                    total_d = res['total_distance']  # 总距离
                    veh_cnt = res['vehicle_count']  # 车辆数
                    worst_d = min(len(r['customers']) for r in res['routes'])  # 节点数最少的路径

                    global_feat = torch.tensor([
                        total_d / 1e4,
                        veh_cnt / len(res['routes']),
                        worst_d / 1e3
                    ], dtype=torch.float, device=device)
                    state_emb = torch.cat([emb, global_feat], dim=-1)  # (embed_dim+3,)
                    q_vals = self.agent_full.qnetwork_local(state_emb.unsqueeze(0)).squeeze(0)  # (action_size,)
                    q_vals = torch.clamp(q_vals, -100, 100)

                    # 打印当前轮、所有动作的 Q 值，便于调试（默认关闭，仅verbose时每10轮打印一次）
                    if self.verbose and (round_idx % 10 == 0):
                        print(f"Round {round_idx}, Q-values: {q_vals.cpu().detach().numpy().round(2)}")
                    q_vals_np = q_vals.detach().cpu().numpy()

                    # 4) ε-greedy 选择动作
                    if random.random() > self.eps:
                        act = int(np.argmax(q_vals_np))
                    else:
                        act = random.randrange(self.action_size)

                    # 5) 执行动作：破坏 + 修复（Full 空间：使用 full 实例的破坏算子）
                    op_d_idx = act // len(self.repair_ops)
                    op_r_idx = act % len(self.repair_ops)
                    destroyed, removed = self.destroy_ops_full[op_d_idx](curr_full)
                    new_full = self.repair_ops[op_r_idx](destroyed, removed, self.instance)
                    # 在解码前进行类型规范化，避免 numpy.int64 等类型潜在问题
                    try:
                        new_full = [int(x) for x in new_full]
                    except Exception:
                        pass
                    # 避免重复解码，这里先不计算 new_full_cost，稍后从 fast 解码结果赋值
                    new_full_cost = None

                    # 6) 组织 DQN 的 (s, a, r, s')
                    state_np = state_emb.detach().cpu().numpy().reshape(1, -1)

                    # 6.1) 解码 next_state 的全局特征（使用 fast 策略）
                    next_feat_f = self.extract_node_features(new_full, space='full')
                    next_emb = self.aggregator(next_feat_f)  # (embed_dim,)

                    res2 = self.decoder.decode_solution(new_full, strategy='fast')
                    total2 = res2['total_distance']
                    veh2 = res2['vehicle_count']
                    worst2 = max(r['distance'] for r in res2['routes'])

                    # 6.2) 计算奖励（距离改进 + 车辆数减少 + 最差路径缩短），并设置 new_full_cost
                    new_full_cost = total2
                    dist_gain = curr_full_cost - new_full_cost
                    veh_gain = float(veh_cnt - veh2)
                    worst_gain = float(worst_d - worst2)
                    reward = dist_gain + 1000.0 * veh_gain + 0.1 * worst_gain
                    global2 = torch.tensor([
                        total2 / 1e4,
                        veh2 / len(res2['routes']),
                        worst2 / 1e3
                    ], dtype=torch.float, device=device)

                    next_state_emb = torch.cat([next_emb, global2], dim=-1)  # (embed_dim+3,)
                    next_state_np = next_state_emb.detach().cpu().numpy().reshape(1, -1)

                    action_np = np.array([[act]])
                    reward_np = np.array([[reward]])
                    done_np = np.array([[0]])

                    self.agent_full.step(
                        state_np, action_np, reward_np, next_state_np, done_np
                    )

                    # 7) 接受新解
                    curr_full = new_full.copy()
                    curr_full_cost = new_full_cost
                    self.eps = max(self.eps * self.eps_decay, self.eps_end)

                    # 8) 更新 local best（带多样性保护）
                    worst_full_idx = int(np.argmax(pop_full_cost))
                    best_full_idx = int(np.argmin(pop_full_cost))
                    current_min_cost = pop_full_cost[best_full_idx]

                    # 多样性保护：避免产生重复的最小值
                    if curr_full_cost < pop_full_cost[worst_full_idx]:
                        # 如果新解等于当前最小值，则不更新（保持多样性）
                        # 如果新解小于当前最小值，则允许更新（找到了更好的解）
                        if curr_full_cost < current_min_cost or round_idx == 1:
                            pop_full[worst_full_idx] = curr_full
                            pop_full_cost[worst_full_idx] = curr_full_cost
                    best_full_history.append(pop_full_cost[best_full_idx])
                    if self.verbose:
                        print(pop_full_cost)
                    
                    # 8.1) 最差路径优化：在每次DQN动作后对当前个体执行最差路径优化
                    if round_idx % 3 == 0:  # 每3轮执行一次最差路径优化
                        try:
                            # 对当前个体执行最差路径优化
                            improved_solution = self.worst_route_optimizer.optimize_worst_route(
                                curr_full, 
                                max_iterations=3  # 在循环内部限制更少的迭代次数以控制计算时间
                            )
                            
                            if improved_solution is not None:
                                improved_cost = self._evaluate_full(improved_solution)
                                
                                # 如果找到改进，更新当前解
                                if improved_cost < curr_full_cost:
                                    old_cost = curr_full_cost
                                    curr_full = improved_solution
                                    curr_full_cost = improved_cost
                                    
                                    # 同时更新种群中对应的个体
                                    if curr_full_cost < pop_full_cost[j]:
                                        pop_full[j] = curr_full.copy()
                                        pop_full_cost[j] = curr_full_cost
                                    
                                    if self.verbose:
                                        print(f"    个体 {j}: 最差路径优化 {old_cost:.2f} -> {improved_cost:.2f}")
                        
                        except Exception as e:
                            if self.verbose:
                                print(f"    个体 {j}: 最差路径优化失败 - {str(e)}")
                # 用 best_full_local 更新 pop_full[j]

            # ---------------------------------------
            # B.1.5) 局部搜索：在Full空间对种群进行局部搜索优化（按频率触发）
            # ---------------------------------------
            # 修复：使用Full空间内部迭代计数器，而不是外层轮次
            if self.enable_local_search and (full_iter_idx % self.local_search_frequency == 0):
                if self.verbose:
                    print(f"[Round {round_idx}, Full迭代 {full_iter_idx+1}] 执行局部搜索...")

                # 对前50%精英执行局部搜索，提升质量
                num_to_search = max(1, (self.pop_size + 1) // 2)
                best_indices = np.argsort(pop_full_cost)[:num_to_search]

                for k in best_indices:
                    try:
                        # 自适应迭代次数：规模越大、迭代数越多（但上限控制在 60）
                        n_cust = len(self.instance.ordinary_customers)
                        ls_iters = max(50, min(60, n_cust // 8))

                        # 使用 AlnsOperators 中的局部搜索方法
                        improved_solution, improved_cost = None, None
                        
                        # 映射局部搜索类型名到实际方法名
                        ls_method_name = self.local_search_type
                        if self.local_search_type == 'adaptive':
                            ls_method_name = 'adaptive_local_search'
                        elif self.local_search_type == 'two_opt':
                            ls_method_name = 'two_opt_local_search'
                        elif self.local_search_type == 'or_opt':
                            ls_method_name = 'or_opt_local_search'
                        elif self.local_search_type == 'swap':
                            ls_method_name = 'swap_local_search'
                        elif self.local_search_type == 'relocate':
                            ls_method_name = 'relocate_local_search'
                        elif self.local_search_type == 'cluster_based':
                            ls_method_name = 'cluster_based_local_search'
                        
                        if hasattr(self.alns_ops, ls_method_name):
                            ls_method = getattr(self.alns_ops, ls_method_name)
                            improved_solution = ls_method(
                                pop_full[k],
                                max_iterations=ls_iters
                            )
                            # 计算改进解的成本
                            improved_cost = self._evaluate_full(improved_solution)
                        else:
                            # 如果找不到指定的局部搜索方法，则跳过
                            if self.verbose:
                                print(f"  警告: 局部搜索类型 '{self.local_search_type}' (映射为 '{ls_method_name}') 未在 AlnsOperators 中找到。跳过个体 {k} 的局部搜索。")
                            continue

                        # 如果局部搜索找到更好的解，则更新（带多样性保护）
                        if improved_cost < pop_full_cost[k]:
                            current_min_cost = min(pop_full_cost)
                            # 改进的多样性保护逻辑：
                            # 1. 如果找到更好的全局最优解，总是更新
                            # 2. 如果改进幅度足够大（超过1%），允许更新
                            # 3. 如果是第一轮，允许更新
                            # 4. 如果当前个体不是最优个体，允许适度改进
                            improvement_ratio = (pop_full_cost[k] - improved_cost) / pop_full_cost[k]
                            is_significant_improvement = improvement_ratio > 0.01  # 改进超过1%
                            is_not_best_individual = pop_full_cost[k] > current_min_cost * 1.05  # 当前个体不是接近最优的

                            if (improved_cost < current_min_cost or  # 找到更好的全局最优解
                                    round_idx == 1 or  # 第一轮
                                    is_significant_improvement or  # 显著改进
                                    is_not_best_individual):  # 非最优个体的适度改进
                                old_cost = pop_full_cost[k]
                                pop_full[k] = improved_solution
                                pop_full_cost[k] = improved_cost
                                if self.verbose:
                                    pass
                                    #print(f"  个体 {k}: 局部搜索改进 {old_cost:.2f} -> {improved_cost:.2f}")
                            else:
                                if self.verbose:
                                    pass
                                    #print(f"  个体 {k}: 局部搜索找到解 {improved_cost:.2f}，但为保持多样性未更新")
                    except Exception as e:
                        if self.verbose:
                            print(f"  个体 {k}: 局部搜索失败 - {str(e)}")
                        continue

            # ---------------------------------------
            # ---------------------------------------
            # B.2) Full->Deep: 使用最优 Full 解动态调整 Deep 空间，并重构 Deep 种群
            # ---------------------------------------
            best_full_idx = int(np.argmin(pop_full_cost))
            best_full_sol = pop_full[best_full_idx]
            
            if self.verbose:
                old_Nd = self.Nd
                old_deep_costs = pop_deep_cost.copy()
                print(f"  [Deep空间重构] 使用最优Full解 (个体{best_full_idx}, 成本{pop_full_cost[best_full_idx]:.2f}) 重构Deep空间")
                print(f"    重构前: Deep节点数={old_Nd}, Deep种群成本={[f'{cost:.2f}' for cost in old_deep_costs]}")
        
            # 1. 使用最优 full 解，基于几何邻近性重新定义 Deep 空间。
            #    这会更新 self.deep_inst, self.Nd, self.destroy_ops_deep 等。
            self.reduce_full_to_deep(best_full_sol, update_deep_space=True, show_visualization=False)
        
            # 2. Deep 空间已改变，必须重构整个 pop_deep 种群以保持一致性。
            #    我们遍历 pop_full，将每个 full 解映射到新的 deep 空间。
            if self.verbose:
                print(f"    重构后: Deep节点数={self.Nd}")
                print(f"  [Deep种群重映射] 将所有Full个体重新映射到新Deep空间...")
            
            for k in range(self.pop_size):
                old_deep_seq = pop_deep[k].copy()
                old_deep_cost = pop_deep_cost[k]
                
                # map_full_to_current_deep 会使用 reduce_full_to_deep 中更新的 self.current_group_id 来映射
                mapped_deep_seq = self.map_full_to_current_deep(pop_full[k])
                pop_deep[k] = mapped_deep_seq
                # _evaluate_deep 会使用更新后的 self.deep_inst
                pop_deep_cost[k] = self._evaluate_deep(mapped_deep_seq)
                
                if self.verbose:
                    print(f"    个体 {k}: {old_deep_seq} (成本{old_deep_cost:.2f}) -> {mapped_deep_seq} (成本{pop_deep_cost[k]:.2f})")
        
            if self.verbose:
                print(f"  [Deep空间重构完成] 新Deep种群成本: {[f'{cost:.2f}' for cost in pop_deep_cost]}")
                cost_change = np.mean(pop_deep_cost) - np.mean(old_deep_costs)
                print(f"    平均成本变化: {cost_change:+.2f}")
            
            # 3.1) 生成节点聚合可视化
            try:
                # 直接使用Deep节点作为聚类簇，不再进行二次聚合
                # 每个Deep节点就是一个聚类，聚类标签就是Deep节点索引
                unique_labels = list(range(self.Nd))  # Deep节点索引作为聚类标签
                cluster_sizes = [len(self.dc.VC_new[label]) for label in unique_labels]
                cluster_info = {
                    'n_clusters': self.Nd,  # 聚类数量等于Deep节点数量
                    'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
                    'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
                    'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0
                }
                
                # 性能指标
                performance_metrics = {
                    'before_cost': np.mean(old_deep_costs),
                    'after_cost': np.mean(pop_deep_cost),
                    'cost_change': cost_change
                }
                
                # 生成可视化
                self.cluster_visualizer.visualize_cluster_aggregation(
                    instance=self.instance,
                    deep_inst=self.deep_inst,
                    cluster_labels=np.array(self.cluster_label),
                    round_idx=round_idx,
                    best_full_solution=best_full_sol,
                    title_suffix=f"成本变化: {cost_change:+.2f}"
                )
                
                # 生成摘要报告
                self.cluster_visualizer.create_summary_report(
                    round_idx=round_idx,
                    original_nodes=len(self.instance.customers),
                    deep_nodes=self.Nd,
                    cluster_info=cluster_info,
                    performance_metrics=performance_metrics
                )
                
            except Exception as e:
                if self.verbose:
                    print(f"  [可视化警告] 生成聚合可视化失败: {str(e)}")
        
            # ===============================
            # 3) 记录并打印本轮最优
            # ===============================
            best_full_idx = int(np.argmin(pop_full_cost))
            best_deep_idx = int(np.argmin(pop_deep_cost))
            best_full_cost = pop_full_cost[best_full_idx]
            best_deep_cost = pop_deep_cost[best_deep_idx]
            best_full_history.append(best_full_cost)

            if self.verbose:
                print(f"[Round {round_idx:3d}] Best Full Cost = {best_full_cost:.2f}, "
                      f"Best Deep Cost = {best_deep_cost:.2f}")
            
            # 多样化策略：检查是否需要重启种群
            if round_idx % 20 == 0:  # 每20轮检查一次
                diversity = self._calculate_population_diversity(pop_full, pop_full_cost)
                if self._should_restart(best_full_cost, diversity):
                    pop_full, pop_full_cost, pop_deep, pop_deep_cost = self._restart_population(
                        pop_full, pop_full_cost, pop_deep, pop_deep_cost)
                    # 重启后重置无改进计数器
                    self.no_improvement_count = 0

        # 返回全局最优 Full 解、成本及历史最优
        if thirty_minute_data is None:
            best_full_idx = int(np.argmin(pop_full_cost))
            thirty_minute_data = {
                'best_sequence': pop_full[best_full_idx],
                'best_cost': pop_full_cost[best_full_idx],
                'cost_history': best_full_history.copy()
            }

            # 返回全局最优 Full 解、成本及历史最优，以及 10 分钟和 30 分钟时的数据
        best_full_idx = int(np.argmin(pop_full_cost))
        return (
            pop_full[best_full_idx],
            pop_full_cost[best_full_idx],
            best_full_history,
            ten_minute_data,
            thirty_minute_data
        )

    def reduce_full_to_deep(self, full_chrom, dist_threshold: float | None = None, use_adaptive_threshold: bool = True,
                            show_visualization: bool = False, update_deep_space: bool = False):
        """
        Full->Deep 小力度降维（按相邻几何距离分组，不再使用既有聚类簇）：
        1) 解码 Full 染色体为多条路径 routes；
        2) 在每条路径内按"相邻欧氏距离 <= 阈值"进行连续分组，得到 groups（保持路径与客户顺序）；
        3) 每个分组作为一个"深度客户节点"，构建新的 deepdata；
        4) 若 update_deep_space=True，则刷新 self.deep_inst / self.Nd / self.init_gen_deep / self.current_VC_new / self.current_group_id；
        5) 返回基于该 full 序列的 Deep 染色体（分组索引序列，顺序即 0..len(groups)-1）。

        :param use_adaptive_threshold: 是否使用自适应阈值（基于路径统计计算），默认True
        """
        # 1) 解码
        res = self.decoder.decode_solution([int(x) for x in full_chrom], strategy='fast')
        routes = res.get('routes', [])

        # 2) 确定阈值：优先级为 dist_threshold > adaptive > default
        if dist_threshold is not None:
            thr = float(dist_threshold)
            threshold_type = "指定阈值"
        elif use_adaptive_threshold:
            thr = self._calculate_adaptive_threshold(routes)
            threshold_type = "自适应阈值"
        else:
            thr = float(self.distance_threshold)
            threshold_type = "默认阈值"

        if self.verbose:
            print(f"\n=== Full->Deep 分组降维 ({threshold_type}={thr:.2f}) ===")
            print(f"原始路径数: {len(routes)}")

        # 3) 按相邻距离分组
        groups = self._group_routes_by_distance(routes, threshold=thr)
        if self.verbose:
            print(f"分组数: {len(groups)}  示例前5组: {groups[:5]}")
        # 3) 构建 deepdata
        deepdata = self._build_deepdata_from_groups(groups)
        # 3.5) 可选可视化（若已有可视化工具，可在此对比原路径与分组，不再粘贴具体代码）
        if show_visualization:
            try:
                visualize_reduction_comparison(routes, groups, self.instance, title=f"路径分组降维对比(阈值={thr})")
            except Exception as e:
                if self.verbose:
                    print(f"可视化失败: {e}")
        # 4) 更新 Deep 空间（可选）
        if update_deep_space:
            self.deep_inst = DeepVRPTWInstance(deepdata)
            self.Nd = len(deepdata['customer'])
            self.init_gen_deep = InitialSolutionGenerator(self.deep_inst)
            # 更新destroy_ops_deep以使用新的deep_inst（优化：增加移除数量和操作多样性）
            # 使用现有的ALNS操作符，适配深度空间的参数
            self.destroy_ops_deep = [
                lambda chrom: self.alns_ops.random_removal(chrom, max(2, len(chrom) // 4)),  # 从1/5增加到1/4
                lambda chrom: self.alns_ops.shaw_removal(chrom, max(2, len(chrom) // 4)),    # 从1/5增加到1/4
                lambda chrom: self.alns_ops.complete_route_removal(chrom, target_routes=1)   # 使用完整路径移除操作
            ]
            self.current_VC_new = groups
            self.current_group_id = {c: i for i, g in enumerate(groups) for c in g}
        # 5) 返回该 full 序列对应的 deep 染色体（分组按顺序一次出现）
        deep_seq = list(range(len(groups)))
        return deep_seq

    # 新增：把任意 Full 序列映射到“当前 Deep 空间”（由分组定义的节点序列）
    def map_full_to_current_deep(self, full_seq):
        """
        使用 self.current_group_id（原客户 -> 组索引）把 Full 序列映射为 Deep 染色体：
        - 按 Full 序列遍历，每遇到一个组索引，若尚未出现则追加；
        - 结束后若仍有未覆盖的组，则按自然顺序补齐，确保 deep 染色体覆盖所有组一次。
        """
        group_id = getattr(self, 'current_group_id', None)
        if group_id is None:
            # 若未建立分组，则退化为原 VC_new 的标签顺序
            vc = getattr(self, 'current_VC_new', self.dc.VC_new)
            seen = set()
            seq = []
            for node in full_seq:
                # 在原 VC_new 中查找所属组
                gid = next((i for i, g in enumerate(vc) if node in g), None)
                if gid is None:
                    continue
                if gid not in seen:
                    seen.add(gid)
                    seq.append(gid)
            # 补齐
            for gid in range(len(vc)):
                if gid not in seen:
                    seq.append(gid)
            return seq
        # 正常路径
        seen = set()
        seq = []
        for node in full_seq:
            gid = group_id.get(node, None)
            if gid is None:
                continue
            if gid not in seen:
                seen.add(gid)
                seq.append(gid)
        # 补齐
        total = len(getattr(self, 'current_VC_new', []))
        for gid in range(total):
            if gid not in seen:
                seq.append(gid)
        return seq

    def _calculate_adaptive_threshold(self, routes):
        """
        基于路径统计信息计算自适应阈值：路径总距离除以客户点数量

        :param routes: 解码后的路径列表
        :return: 计算得到的自适应阈值
        """
        if not routes:
            return self.distance_threshold  # 如果没有路径，返回默认阈值

        coords = self.instance.ordinary_customers

        def dist(u, v):
            cu, cv = coords[u], coords[v]
            dx = cu['xcoord'] - cv['xcoord']
            dy = cu['ycoord'] - cv['ycoord']
            return float((dx * dx + dy * dy) ** 0.5)

        total_distance = 0.0
        total_customers = 0
        depot = self.instance.customers[0]

        for route in routes:
            customers = route.get('customers', [])
            if not customers:
                continue

            total_customers += len(customers)

            # 计算路径总距离：仓库->第一个客户->...->最后一个客户->仓库
            route_distance = 0.0

            # 仓库到第一个客户
            first_customer = coords[customers[0]]
            dx = first_customer['xcoord'] - depot['xcoord']
            dy = first_customer['ycoord'] - depot['ycoord']
            route_distance += (dx * dx + dy * dy) ** 0.5

            # 客户间距离
            for i in range(len(customers) - 1):
                route_distance += dist(customers[i], customers[i + 1])

            # 最后一个客户回到仓库
            last_customer = coords[customers[-1]]
            dx = depot['xcoord'] - last_customer['xcoord']
            dy = depot['ycoord'] - last_customer['ycoord']
            route_distance += (dx * dx + dy * dy) ** 0.5

            total_distance += route_distance

        if total_customers == 0:
            return self.distance_threshold

        # 计算平均每个客户的路径距离作为阈值
        adaptive_threshold = total_distance / total_customers * 0.5

        if self.verbose:
            print(
                f"自适应阈值计算: 总距离={total_distance:.2f}, 总客户数={total_customers}, 阈值={adaptive_threshold:.2f}")

        return adaptive_threshold

    # 新增：按相邻几何距离在单条路径内分组，不依赖原聚类
    def _group_routes_by_distance(self, routes, threshold: float):
        """
        将解码得到的多条路径，按每条路径内“相邻客户欧氏距离 <= threshold”进行连续分组。
        返回按照路径顺序拼接的分组列表，每个分组是原空间客户索引列表。
        """
        coords = self.instance.ordinary_customers

        def dist(u, v):
            cu, cv = coords[u], coords[v]
            dx = cu['xcoord'] - cv['xcoord']
            dy = cu['ycoord'] - cv['ycoord']
            return float((dx * dx + dy * dy) ** 0.5)

        groups = []
        for route in routes:
            customers = route.get('customers', [])
            if not customers:
                continue
            current = [customers[0]]
            for i in range(len(customers) - 1):
                if dist(customers[i], customers[i + 1]) <= threshold:
                    current.append(customers[i + 1])
                else:
                    groups.append(current.copy())
                    current = [customers[i + 1]]
            if current:
                groups.append(current)
        return groups

    # 新增：由分组构造新的 deepdata（组即“深度客户节点”），并返回 VC_new=groups
    def _build_deepdata_from_groups(self, groups):
        """
        基于分组 groups 动态构建 Deep 空间实例所需的 deepdata：
        - 每个分组聚合为一个“深度客户节点”：坐标取均值，需求为和，服务时间=组内服务时间和+组内相邻行驶时间，
          ReadyTime 取 max(首客户的 ready_time, 仓库到首客户的行驶时间)，DueDate 取末客户 due_date。
        - VC_new 直接等于分组列表（原客户索引列表）。
        """
        inst = self.instance
        D = inst.distance_matrix  # 含仓库在 [0,*]
        depot = inst.warehouse
        deepdata = {
            'vehicle': inst.vehicle_info,
            'depot': {
                'xcoord': depot['xcoord'],
                'ycoord': depot['ycoord'],
                'due_date': depot['due_date']
            },
            'customer': [],
            'VC_new': groups
        }
        custs = inst.ordinary_customers
        for group in groups:
            xs = [custs[i]['xcoord'] for i in group]
            ys = [custs[i]['ycoord'] for i in group]
            ds = [custs[i]['demand'] for i in group]
            sts = [custs[i]['service_time'] for i in group]
            rn = len(group)
            
            # 计算聚合后的时间窗参数
            if rn == 1:
                # 单个客户的情况
                i = group[0]
                ready_time = custs[i]['ready_time']
                due_date = custs[i]['due_date']
                service_time = custs[i]['service_time']
            else:
                # 多个客户聚合的情况
                first_customer = group[0]
                second_customer = group[1]
                
                # 1. 开始时间 = 第一个客户的ReadyTime
                ready_time = custs[first_customer]['ready_time']
                
                # 2. 截止时间 = 第二个客户的截止时间 - 第一个客户到第二个客户的距离
                travel_time_1_to_2 = float(D[first_customer + 1, second_customer + 1])
                due_date = custs[second_customer]['due_date'] - travel_time_1_to_2
                
                # 3. 服务时间 = 完成整个子路径所需的总时间
                # 包括所有客户的服务时间 + 客户间的行驶时间
                service_time = 0
                for j in range(rn):
                    customer_idx = group[j]
                    service_time += custs[customer_idx]['service_time']
                    
                    # 添加到下一个客户的行驶时间
                    if j < rn - 1:
                        next_customer_idx = group[j + 1]
                        service_time += float(D[customer_idx + 1, next_customer_idx + 1])
            
            deepdata['customer'].append({
                'x': float(np.mean(xs)) if rn > 0 else 0.0,
                'y': float(np.mean(ys)) if rn > 0 else 0.0,
                'demand': int(sum(ds)),
                'ServiceTime': float(service_time),
                'ReadyTime': int(ready_time),
                'DueDate': int(due_date)
            })
        return deepdata

    def _calculate_population_diversity(self, population, costs):
        """计算种群多样性"""
        if len(population) < 2:
            return 1.0
        
        total_distance = 0
        count = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                # 计算两个解之间的汉明距离（不同位置的比例）
                seq1, seq2 = population[i], population[j]
                if len(seq1) == len(seq2):
                    diff_count = sum(1 for a, b in zip(seq1, seq2) if a != b)
                    distance = diff_count / len(seq1)
                    total_distance += distance
                    count += 1
        
        return total_distance / count if count > 0 else 0.0

    def _restart_population(self, pop_full, pop_full_cost, pop_deep, pop_deep_cost):
        """重启种群，保留精英个体"""
        if not self.enable_restart:
            return pop_full, pop_full_cost, pop_deep, pop_deep_cost
        
        # 计算保留的精英个体数量
        elite_count = max(1, int(self.pop_size * self.elite_ratio))
        
        # Full空间：保留精英，重新生成其他个体
        elite_indices = np.argsort(pop_full_cost)[:elite_count]
        new_pop_full = [pop_full[i] for i in elite_indices]
        new_pop_full_cost = [pop_full_cost[i] for i in elite_indices]
        
        # 生成新的个体填充种群
        for _ in range(self.pop_size - elite_count):
            sol_full = self.init_gen.generate_random_solution()
            cost_full = self._evaluate_full(sol_full)
            new_pop_full.append(sol_full)
            new_pop_full_cost.append(cost_full)
        
        # Deep空间：保留精英，重新生成其他个体
        elite_indices = np.argsort(pop_deep_cost)[:elite_count]
        new_pop_deep = [pop_deep[i] for i in elite_indices]
        new_pop_deep_cost = [pop_deep_cost[i] for i in elite_indices]
        
        # 生成新的个体填充种群
        for _ in range(self.pop_size - elite_count):
            sol_deep = random.sample(range(self.Nd), self.Nd)
            cost_deep = self._evaluate_deep(sol_deep)
            new_pop_deep.append(sol_deep)
            new_pop_deep_cost.append(cost_deep)
        
        if self.verbose:
            print(f"种群重启：保留{elite_count}个精英个体，重新生成{self.pop_size - elite_count}个个体")
        
        return new_pop_full, new_pop_full_cost, new_pop_deep, new_pop_deep_cost

    def _should_restart(self, current_best_cost, diversity):
        """判断是否应该重启种群"""
        if not self.enable_restart:
            return False
        
        # 检查是否有改进
        if current_best_cost < self.last_best_cost:
            self.no_improvement_count = 0
            self.last_best_cost = current_best_cost
        else:
            self.no_improvement_count += 1
        
        # 重启条件：无改进轮数超过阈值 或 种群多样性过低
        should_restart = (self.no_improvement_count >= self.restart_threshold or 
                         diversity < self.diversity_threshold)
        
        return should_restart

    # ===============================
    # 遗传算法操作函数
    # ===============================
    
    def ga_tournament_selection(self, population, costs, tournament_size=None):
        """锦标赛选择"""
        if tournament_size is None:
            tournament_size = self.ga_tournament_size
            
        # 随机选择tournament_size个个体
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_costs = [costs[i] for i in tournament_indices]
        
        # 选择成本最低的个体
        winner_idx = tournament_indices[np.argmin(tournament_costs)]
        return population[winner_idx].copy()
    
    def ga_order_crossover(self, parent1, parent2):
        """顺序交叉（Order Crossover, OX）- 适用于排列编码"""
        if random.random() > self.ga_crossover_rate:
            return parent1.copy(), parent2.copy()
        
        size = len(parent1)
        # 随机选择两个交叉点
        start, end = sorted(random.sample(range(size), 2))
        
        # 创建子代1
        child1 = [-1] * size
        child1[start:end] = parent1[start:end]
        
        # 从parent2中按顺序填充剩余位置
        p2_filtered = [x for x in parent2 if x not in child1[start:end]]
        j = 0
        for i in range(size):
            if child1[i] == -1:
                child1[i] = p2_filtered[j]
                j += 1
        
        # 创建子代2
        child2 = [-1] * size
        child2[start:end] = parent2[start:end]
        
        # 从parent1中按顺序填充剩余位置
        p1_filtered = [x for x in parent1 if x not in child2[start:end]]
        j = 0
        for i in range(size):
            if child2[i] == -1:
                child2[i] = p1_filtered[j]
                j += 1
                
        return child1, child2
    
    def ga_swap_mutation(self, individual):
        """交换变异"""
        if random.random() > self.ga_mutation_rate:
            return individual.copy()
        
        mutated = individual.copy()
        # 随机选择两个位置进行交换
        i, j = random.sample(range(len(mutated)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        
        return mutated
    
    def ga_evolve_population(self, population, costs, space='full'):
        """对种群执行一轮遗传算法进化"""
        pop_size = len(population)
        new_population = []
        new_costs = []
        
        # 精英保留策略：保留最好的个体
        elite_indices = np.argsort(costs)[:self.ga_elite_count]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
            new_costs.append(costs[idx])
        
        # 生成剩余个体
        while len(new_population) < pop_size:
            # 选择父代
            parent1 = self.ga_tournament_selection(population, costs)
            parent2 = self.ga_tournament_selection(population, costs)
            
            # 交叉
            child1, child2 = self.ga_order_crossover(parent1, parent2)
            
            # 变异
            child1 = self.ga_swap_mutation(child1)
            child2 = self.ga_swap_mutation(child2)
            
            # 评估子代并添加到新种群
            for child in [child1, child2]:
                if len(new_population) < pop_size:
                    if space == 'deep':
                        child_cost = self._evaluate_deep(child)
                    else:
                        child_cost = self._evaluate_full(child)
                    
                    new_population.append(child)
                    new_costs.append(child_cost)
        
        return new_population, new_costs
    
    def get_elite_individuals(self, population, costs, count=None):
        """获取种群中最好的几个个体"""
        if count is None:
            count = self.ga_elite_count
            
        elite_indices = np.argsort(costs)[:count]
        elite_individuals = [population[i] for i in elite_indices]
        elite_costs = [costs[i] for i in elite_indices]
        
        return elite_individuals, elite_costs, elite_indices
    
    def select_elite_and_random_individuals(self, population, costs, elite_count=10, random_count=10):
        """选择精英个体和随机个体用于GA操作
        
        Args:
            population: 种群
            costs: 个体成本
            elite_count: 精英个体数量
            random_count: 随机个体数量
            
        Returns:
            selected_individuals: 选中的个体列表
            selected_costs: 选中个体的成本列表
            selected_indices: 选中个体的原始索引列表
        """
        pop_size = len(population)
        
        # 1. 选择精英个体（最好的elite_count个）
        elite_indices = np.argsort(costs)[:elite_count]
        
        # 2. 从剩余个体中随机选择random_count个
        remaining_indices = list(range(pop_size))
        for idx in elite_indices:
            remaining_indices.remove(idx)
        
        # 确保有足够的剩余个体进行随机选择
        actual_random_count = min(random_count, len(remaining_indices))
        random_indices = np.random.choice(remaining_indices, actual_random_count, replace=False)
        
        # 3. 合并选中的个体
        selected_indices = np.concatenate([elite_indices, random_indices])
        selected_individuals = [population[i] for i in selected_indices]
        selected_costs = [costs[i] for i in selected_indices]
        
        return selected_individuals, selected_costs, selected_indices
    
    def ga_evolve_selected_individuals(self, population, costs, space='full'):
        """对种群中选中的个体执行GA操作，然后更新回原种群
        
        Args:
            population: 完整种群列表
            costs: 完整种群成本列表
            space: 空间类型 ('full' 或 'deep')
            
        Returns:
            updated_population: 更新后的完整种群列表
            updated_costs: 更新后的完整种群成本列表
        """
        # 1. 选择10个精英个体和10个随机个体
        selected_individuals, selected_costs, selected_indices = self.select_elite_and_random_individuals(
            population, costs, elite_count=10, random_count=10)
        
        selected_size = len(selected_individuals)
        new_selected_population = []
        new_selected_costs = []
        
        # 2. 精英保留策略：保留最好的2个个体
        elite_indices = np.argsort(selected_costs)[:2]
        for idx in elite_indices:
            new_selected_population.append(selected_individuals[idx].copy())
            new_selected_costs.append(selected_costs[idx])
        
        # 3. 生成剩余个体
        while len(new_selected_population) < selected_size:
            # 从选中的个体中选择父代
            parent1 = self.ga_tournament_selection(selected_individuals, selected_costs)
            parent2 = self.ga_tournament_selection(selected_individuals, selected_costs)
            
            # 交叉
            child1, child2 = self.ga_order_crossover(parent1, parent2)
            
            # 变异
            child1 = self.ga_swap_mutation(child1)
            child2 = self.ga_swap_mutation(child2)
            
            # 评估子代并添加到新种群
            for child in [child1, child2]:
                if len(new_selected_population) < selected_size:
                    if space == 'deep':
                        child_cost = self._evaluate_deep(child)
                    else:
                        child_cost = self._evaluate_full(child)
                    
                    new_selected_population.append(child)
                    new_selected_costs.append(child_cost)
        
        # 4. 将进化后的个体更新回原种群
        updated_population = population.copy()
        updated_costs = costs.copy()
        
        for i, original_idx in enumerate(selected_indices):
            updated_population[original_idx] = new_selected_population[i]
            updated_costs[original_idx] = new_selected_costs[i]
        
        return updated_population, updated_costs
    
    def apply_local_search_operations(self, population, costs, space='full', max_iterations=None):
        """
        对种群中的个体应用局部搜索操作
        包括：2-opt、or-opt、swap、relocate、2-opt*、cross-exchange、3-opt、VNS等
        
        Args:
            population: 种群列表
            costs: 种群成本列表
            space: 空间类型 ('full' 或 'deep')
            max_iterations: 最大迭代次数，如果为None则使用自适应参数
            
        Returns:
            improved_population: 改进后的种群列表
            improved_costs: 改进后的种群成本列表
        """
        improved_population = []
        improved_costs = []
        
        # 定义局部搜索操作列表
        local_search_ops = [
            self.alns_ops.two_opt_local_search,
            self.alns_ops.or_opt_local_search,
            self.alns_ops.swap_local_search,
            self.alns_ops.relocate_local_search,
            self.alns_ops.two_opt_star_local_search,
            self.alns_ops.cross_exchange_local_search,
            self.alns_ops.three_opt_local_search,
            self.alns_ops.variable_neighborhood_search
        ]
        
        for i, individual in enumerate(population):
            current_individual = individual.copy()
            current_cost = costs[i]
            
            # 对每个个体应用多种局部搜索操作
            for op in local_search_ops:
                try:
                    # 应用局部搜索操作
                    improved_individual = op(current_individual, max_iterations=max_iterations)
                    
                    # 评估改进后的个体
                    if space == 'deep':
                        improved_cost = self._evaluate_deep(improved_individual)
                    else:
                        improved_cost = self._evaluate_full(improved_individual)
                    
                    # 如果有改进，则更新当前个体
                    if improved_cost < current_cost:
                        current_individual = improved_individual
                        current_cost = improved_cost
                        
                except Exception as e:
                    # 如果局部搜索操作失败，继续下一个操作
                    if self.verbose:
                        print(f"局部搜索操作 {op.__name__} 失败: {e}")
                    continue
            
            improved_population.append(current_individual)
            improved_costs.append(current_cost)
        
        return improved_population, improved_costs
    
    def apply_adaptive_local_search(self, population, costs, space='full'):
        """
        对种群应用自适应局部搜索
        
        Args:
            population: 种群列表
            costs: 种群成本列表
            space: 空间类型 ('full' 或 'deep')
            
        Returns:
            improved_population: 改进后的种群列表
            improved_costs: 改进后的种群成本列表
        """
        improved_population = []
        improved_costs = []
        
        for i, individual in enumerate(population):
            try:
                # 使用自适应局部搜索
                improved_individual = self.alns_ops.adaptive_local_search(individual)
                
                # 评估改进后的个体
                if space == 'deep':
                    improved_cost = self._evaluate_deep(improved_individual)
                else:
                    improved_cost = self._evaluate_full(improved_individual)
                
                improved_population.append(improved_individual)
                improved_costs.append(improved_cost)
                
            except Exception as e:
                # 如果自适应局部搜索失败，保留原个体
                if self.verbose:
                    print(f"自适应局部搜索失败: {e}")
                improved_population.append(individual.copy())
                improved_costs.append(costs[i])
        
        return improved_population, improved_costs