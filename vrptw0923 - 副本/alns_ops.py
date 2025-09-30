import numpy as np
from heapq import heappush, heappop
from collections import defaultdict
import random
import math
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple
from scipy.spatial.distance import pdist, squareform
from vrptw_instance import VRPTWInstance
from decoder import VRPTWDecoder

class AlnsOperators:
    def __init__(self, instance: VRPTWInstance, random_state):
        self.instance = instance
        self.random_state = random_state
        self.num_customers = len(instance.ordinary_customers)

        all_nodes_data = [instance.warehouse] + instance.ordinary_customers
        self.nodes = np.array([[c['xcoord'], c['ycoord']] for c in all_nodes_data])
        self.demands = np.array([c['demand'] for c in all_nodes_data])
        self.time_windows = np.array([[c['ready_time'], c['due_date']] for c in all_nodes_data])
        self.service_times = np.array([c['service_time'] for c in all_nodes_data])

        # 建立客户编号和索引之间的映射，兼容0-based索引和原始cust_no编号
        self.index_to_cust_no = [cust['cust_no'] for cust in instance.ordinary_customers]
        self.cust_no_to_index = {cust_no: idx for idx, cust_no in enumerate(self.index_to_cust_no)}

        self.vehicle_capacity = instance.vehicle_info['capacity']
        self.dist_matrix = instance.distance_matrix

        # Shaw Removal relatedness
        self.relatedness_weights = np.array([9, 3, 2, 5, 0.5])
        self.relatedness_matrix = np.full((self.num_customers + 1, self.num_customers + 1), -1.0)
        self.positions = np.zeros((self.num_customers + 1, 2), dtype=int)
        self.routes_by_customer = np.zeros(self.num_customers + 1, dtype=int)

        # Pre-calculate distances and time window similarities
        self.norm_dist_matrix = self.dist_matrix / self.dist_matrix.max()
        self.norm_tw_matrix = np.zeros_like(self.dist_matrix)
        for i in range(self.num_customers + 1):
            for j in range(self.num_customers + 1):
                self.norm_tw_matrix[i, j] = abs(self.time_windows[i, 0] - self.time_windows[j, 0]) + \
                                            abs(self.time_windows[i, 1] - self.time_windows[j, 1])
        self.norm_tw_matrix /= self.norm_tw_matrix.max()
        self.norm_demand_matrix = np.zeros_like(self.dist_matrix)
        for i in range(self.num_customers + 1):
            for j in range(self.num_customers + 1):
                self.norm_demand_matrix[i, j] = abs(self.demands[i] - self.demands[j])
        self.norm_demand_matrix /= self.norm_demand_matrix.max()

        self.decoder = VRPTWDecoder(instance)
        
        # 预计算邻居列表
        # 预计算每个客户的邻居列表（使用0-based客户索引）
        self._neighbor_lists = [[] for _ in range(self.num_customers)]
        if self.dist_matrix is not None:
            for cust_idx in range(self.num_customers):
                # 跳过仓库列，仅保留客户之间的距离
                row = self.dist_matrix[cust_idx + 1, 1:]
                neighbor_order = np.argsort(row)

                neighbors = []
                for neighbor_idx in neighbor_order:
                    if neighbor_idx == cust_idx:
                        continue  # 跳过自身
                    neighbors.append(int(neighbor_idx))
                    if len(neighbors) >= 20:
                        break

                self._neighbor_lists[cust_idx] = neighbors

    def _normalize_customer_index(self, customer: int) -> Tuple[Optional[int], bool]:
        """将外部客户标识转换为内部0-based索引。

        返回 (索引, 是否使用原始cust_no编号)。
        """
        if 0 <= customer < self.num_customers:
            return customer, False
        mapped = self.cust_no_to_index.get(customer)
        if mapped is not None:
            return mapped, True
        return None, False

    def _neighbor_candidates(self, customer: int, limit: Optional[int] = None) -> list[int]:
        """获取与给定客户相邻的候选客户列表，保持输入编号风格。"""
        norm_idx, used_cust_no = self._normalize_customer_index(customer)
        if norm_idx is None or norm_idx >= len(self._neighbor_lists):
            return []

        neighbors = self._neighbor_lists[norm_idx]
        if limit is not None:
            neighbors = neighbors[:limit]

        if used_cust_no:
            return [self.index_to_cust_no[n_idx] for n_idx in neighbors]
        return neighbors[:]

    def _candidate_positions(self, chromosome: List[int], customer: int, max_positions: int = 40) -> List[int]:
        """生成插入候选位置，兼容不同编号体系并自动回退。"""
        if max_positions is not None and max_positions <= 0:
            max_positions = None

        n = len(chromosome)
        if n == 0:
            return [0]

        # 记录各客户在染色体中的出现位置，兼容字典结构
        position_lookup: Dict[int, List[int]] = {}

        def _extract_customer_id(gene) -> Optional[int]:
            if isinstance(gene, dict):
                for key in ("customer", "cust_no", "id", "idx"):
                    value = gene.get(key)
                    if isinstance(value, int):
                        return value
                return None
            return gene if isinstance(gene, int) else None

        for idx, gene in enumerate(chromosome):
            cust_id = _extract_customer_id(gene)
            if cust_id is None:
                continue
            position_lookup.setdefault(cust_id, []).append(idx)

        neighbor_customers = self._neighbor_candidates(customer, limit=max_positions)
        candidate_positions: Set[int] = set()

        for neighbor in neighbor_customers:
            for pos in position_lookup.get(neighbor, []):
                candidate_positions.add(pos)
                candidate_positions.add(pos + 1)

        # 如果没有找到相邻节点，退化为全局扫描
        if not candidate_positions:
            candidate_positions.update(range(n + 1))

        # 始终允许头尾插入
        candidate_positions.add(0)
        candidate_positions.add(n)

        ordered_positions = sorted(pos for pos in candidate_positions if 0 <= pos <= n)
        if max_positions is not None:
            ordered_positions = ordered_positions[:max_positions]
        return ordered_positions

    def _random_choice(self, items, probabilities=None):
        """带兼容性的随机选择，支持Python random与NumPy随机状态。"""
        if not items:
            raise ValueError("Cannot choose from an empty sequence")

        if probabilities is not None:
            probs = np.array(probabilities, dtype=float)
            if probs.ndim != 1 or len(probs) != len(items):
                raise ValueError("Probabilities must match the number of items")
            total = probs.sum()
            if total <= 0:
                probabilities = None
            else:
                probs = probs / total
                probabilities = probs

        if probabilities is None:
            # 简单等概率选择
            if hasattr(self.random_state, "choice"):
                try:
                    return self.random_state.choice(items)
                except TypeError:
                    pass
            return random.choice(list(items))

        # 带权重的选择
        if hasattr(self.random_state, "choice"):
            try:
                return self.random_state.choice(items, p=probabilities)
            except TypeError:
                pass

        cumulative = np.cumsum(probabilities)
        rnd = self.random_state.random() if hasattr(self.random_state, "random") else random.random()
        idx = int(np.searchsorted(cumulative, rnd, side="right"))
        idx = min(idx, len(items) - 1)
        return items[idx]

    def _solution_to_metric(self, chromosome: list[int]) -> tuple[int, float]:
        """将解（染色体）转换为评估指标（车辆数，总距离）"""
        if not chromosome:
            return (999, 99999.9)
        res = self.decoder.decode_solution(chromosome, "fast")
        return (res.get('vehicle_count', 999), res.get('total_distance', 99999.9))

    def _lex_better(self, met1: tuple[int, float], met2: tuple[int, float]) -> bool:
        """比较两个指标，判断 met1 是否在字典序上优于 met2"""
        if met1[0] < met2[0]:
            return True
        if met1[0] == met2[0] and met1[1] < met2[1]:
            return True
        return False

    def _route_distance(self, route: list[int]) -> float:
        """计算单条路径的总距离 (包含往返仓库)"""
        dist = 0.0
        prev = 0
        for cust in route:
            dist += self.dist_matrix[prev, cust + 1]
            prev = cust + 1
        dist += self.dist_matrix[prev, 0]
        return dist

    def greedy_repair(self, destroyed_chromosome: list[int], removed_customers: list[int], K: int = 30) -> list[int]:
        """
        贪心修复算子.

        Args:
            destroyed_chromosome: 破坏后的染色体.
            removed_customers: 被移除的客户列表.
            K: 每次插入时评估的最佳位置数量.

        Returns:
            修复后的染色体.
        """
        chromosome = destroyed_chromosome[:]
        
        for cust in removed_customers:
            best_pos = -1
            best_cost = float('inf')
            
            # 寻找最佳插入位置
            for pos in range(len(chromosome) + 1):
                temp_chrom = chromosome[:pos] + [cust] + chromosome[pos:]
                res = self.decoder.decode_solution(temp_chrom, strategy='fast')
                if res['feasible']:
                    cost = (res['vehicle_count'], res['total_distance'])
                    if self._lex_better(cost, (99, best_cost)):
                        best_cost = cost[1]
                        best_pos = pos

            if best_pos != -1:
                chromosome.insert(best_pos, cust)
            else:
                # 如果找不到可行位置，则追加到末尾 (可能导致解不可行)
                chromosome.append(cust)
                
        return chromosome

    def regret_insertion(self, destroyed_chromosome: list[int], removed_customers: list[int], k: int = 3, positions_per_node: int = 40, fallback_greedy: bool = True) -> list[int]:
        """
        Regret-k 修复算子.

        Args:
            destroyed_chromosome: 破坏后的染色体.
            removed_customers: 被移除的客户列表.
            k: Regret值计算中考虑的最佳插入位置数量.
            positions_per_node: 为每个客户评估的最大插入位置数.
            fallback_greedy: 如果失败，是否回退到贪心修复.

        Returns:
            修复后的染色体.
        """
        chromosome = destroyed_chromosome[:]
        unassigned = removed_customers[:]
        
        while unassigned:
            best_cust = -1
            max_regret = -float('inf')
            
            # 为每个未分配的客户计算regret值
            for cust in unassigned:
                costs = []
                # 评估插入成本
                for pos in range(len(chromosome) + 1):
                    temp_chrom = chromosome[:pos] + [cust] + chromosome[pos:]
                    res = self.decoder.decode_solution(temp_chrom, strategy='fast')
                    if res['feasible']:
                        costs.append(res['total_distance'])
                
                if not costs:
                    continue

                costs.sort()
                regret = sum(costs[i] - costs[0] for i in range(1, min(k, len(costs))))
                
                if regret > max_regret:
                    max_regret = regret
                    best_cust = cust

            if best_cust != -1:
                # 插入具有最大regret值的客户
                best_pos = -1
                min_cost = float('inf')
                for pos in range(len(chromosome) + 1):
                    temp_chrom = chromosome[:pos] + [best_cust] + chromosome[pos:]
                    res = self.decoder.decode_solution(temp_chrom, strategy='fast')
                    if res['feasible'] and res['total_distance'] < min_cost:
                        min_cost = res['total_distance']
                        best_pos = pos
                
                if best_pos != -1:
                    chromosome.insert(best_pos, best_cust)
                    unassigned.remove(best_cust)
                else:
                    # 如果找不到位置，则停止
                    break
            else:
                # 如果没有客户可以插入，则停止
                break

        # 如果仍有未分配的客户，使用贪心修复
        if unassigned and fallback_greedy:
            chromosome = self.greedy_repair(chromosome, unassigned)
            
        return chromosome

    def random_removal(self, chromosome: list[int], remove_count: int) -> tuple[list[int], list[int]]:
        """
        随机移除算子.

        Args:
            chromosome: 当前染色体.
            remove_count: 要移除的客户数量.

        Returns:
            (破坏后的染色体, 被移除的客户列表).
        """
        if not chromosome or remove_count <= 0:
            return chromosome[:], []
        
        k_remove = min(len(chromosome), remove_count)
        if hasattr(self.random_state, "sample"):
            removed_customers = self.random_state.sample(chromosome, k_remove)
        else:
            removed_customers = random.sample(chromosome, k_remove)
        new_chromosome = [c for c in chromosome if c not in removed_customers]
        
        return new_chromosome, removed_customers

    def shaw_removal(self, chromosome: list[int], remove_count: int, shaw_lambda: float = 1.2, marginal_weight: float = 0.4, route_weight: float = 0.3, urgency_weight: float = 0.2, conflict_weight: float = 0.1) -> tuple[list[int], list[int]]:
        """
        改进的Shaw移除算子 (基于多种评价策略).

        Args:
            chromosome: 当前染色体.
            remove_count: 移除数量.
            shaw_lambda: 随机化参数.
            marginal_weight: 边际成本权重.
            route_weight: 路径依赖强度权重.
            urgency_weight: 时间窗紧迫度权重.
            conflict_weight: 冲突性权重.

        Returns:
            (破坏后的染色体, 被移除的客户列表).
        """
        k_remove = max(1, min(len(chromosome), remove_count))
        
        # 解码以获取路径信息
        result = self.decoder.decode_solution(chromosome, strategy='detailed')
        routes = result.get('routes', [])
        total_cost = result.get('total_distance', 0)
        
        # 构建客户到路径的映射
        customer_to_route = {}
        customer_to_position = {}
        
        for route_idx, route_info in enumerate(routes):
            route = route_info.get('customers', [])
            for pos, customer in enumerate(route):
                customer_to_route[customer] = route_idx
                customer_to_position[customer] = pos
        
        if not chromosome:
            return [], []

        # 智能选择起始客户
        route_lengths = [len(route_info.get('customers', [])) for route_info in routes]
        weighted_customers = [customer for customer in chromosome if customer in customer_to_route for _ in range(route_lengths[customer_to_route[customer]])]
        
        if not weighted_customers:
            weighted_customers = chromosome
            
        removed = [self._random_choice(weighted_customers)]
        
        while len(removed) < k_remove:
            candidates = [c for c in chromosome if c not in removed]
            if not candidates:
                break
            
            scores = {}
            
            for node in candidates:
                node_route_idx = customer_to_route.get(node, -1)
                if node_route_idx == -1:
                    continue
                
                # 1. 边际成本 (Marginal Cost)
                # 计算移除该节点后的成本减少
                route = routes[node_route_idx].get('customers', [])
                pos = customer_to_position.get(node, -1)
                
                if pos > 0 and pos < len(route) - 1:
                    prev_node = route[pos - 1]
                    next_node = route[pos + 1]
                    
                    # 计算当前路径段成本
                    current_segment_cost = (self.dist_matrix[prev_node + 1, node + 1] + 
                                           self.dist_matrix[node + 1, next_node + 1])
                    
                    # 计算移除后的路径段成本
                    new_segment_cost = self.dist_matrix[prev_node + 1, next_node + 1]
                    
                    # 边际成本 = 当前成本 - 移除后成本 (值越大表示移除后节省越多)
                    marginal_cost = current_segment_cost - new_segment_cost
                else:
                    # 如果是路径的第一个或最后一个节点
                    if pos == 0 and len(route) > 1:
                        next_node = route[1]
                        marginal_cost = self.dist_matrix[0, node + 1] + self.dist_matrix[node + 1, next_node + 1] - self.dist_matrix[0, next_node + 1]
                    elif pos == len(route) - 1 and len(route) > 1:
                        prev_node = route[pos - 1]
                        marginal_cost = self.dist_matrix[prev_node + 1, node + 1] + self.dist_matrix[node + 1, 0] - self.dist_matrix[prev_node + 1, 0]
                    else:
                        # 单节点路径
                        marginal_cost = self.dist_matrix[0, node + 1] + self.dist_matrix[node + 1, 0]
                
                # 归一化边际成本 (值越大表示更应该移除)
                max_possible_cost = max(self.dist_matrix.max() * 2, 1.0)
                normalized_marginal_cost = marginal_cost / max_possible_cost
                
                # 2. 路径依赖强度 (Route Contribution)
                # 计算节点在路径中的位置对整体结构的影响
                if pos > 0 and pos < len(route) - 1:
                    prev_node = route[pos - 1]
                    next_node = route[pos + 1]
                    
                    # 计算节点两侧邻居的直接距离差
                    # impact = d(prev,node) + d(node,next) - d(prev,next)
                    # 值越大表示该节点"打断"了两个本应相邻的点
                    route_impact = (self.dist_matrix[prev_node + 1, node + 1] + 
                                   self.dist_matrix[node + 1, next_node + 1] - 
                                   self.dist_matrix[prev_node + 1, next_node + 1])
                    
                    # 归一化路径影响 (值越大表示更应该移除)
                    normalized_route_impact = min(route_impact / max_possible_cost, 1.0)
                else:
                    # 端点的路径依赖强度较低
                    normalized_route_impact = 0.3
                
                # 3. 时间窗紧迫度 (Urgency/Slackness)
                node_ready, node_due = self.time_windows[node + 1]
                service_time = 0  # 假设服务时间为0，如果有服务时间数据可以添加
                
                # 计算时间窗口松弛度 (slack = due - ready - service)
                # 值越小表示时间约束越紧
                time_slack = node_due - node_ready - service_time
                
                # 归一化时间紧迫度 (值越大表示更紧迫，更应该移除)
                max_time_window = max([(tw[1] - tw[0]) for tw in self.time_windows[1:]])
                normalized_urgency = 1.0 - min(time_slack / max(max_time_window, 1.0), 1.0)
                
                # 4. 冲突性/不兼容性 (Infeasibility Impact)
                # 计算与其他节点的冲突程度
                conflict_score = 0.0
                count_conflicts = 0
                
                for route_idx, route_info in enumerate(routes):
                    route_customers = route_info.get('customers', [])
                    for other_node in route_customers:
                        if other_node == node or other_node in removed:
                            continue
                            
                        # 计算时间窗口冲突
                        other_ready, other_due = self.time_windows[other_node + 1]
                        time_conflict = max(0, 1.0 - max(0, min(node_due, other_due) - max(node_ready, other_ready)) / max(max_time_window, 1.0))
                        
                        # 累加冲突得分
                        conflict_score += time_conflict
                        count_conflicts += 1
                
                # 归一化冲突得分
                normalized_conflict = conflict_score / max(count_conflicts, 1)
                
                # 综合评分 (各项指标加权求和)
                # 值越大表示更应该移除该节点
                node_score = (marginal_weight * normalized_marginal_cost + 
                             route_weight * normalized_route_impact + 
                             urgency_weight * normalized_urgency + 
                             conflict_weight * normalized_conflict)
                
                # 添加随机因子
                random_factor = self.random_state.random() ** shaw_lambda
                scores[node] = node_score * random_factor
            
            if scores:
                sorted_candidates = sorted(scores.items(), key=lambda x: x[1])
                top_candidates = sorted_candidates[:max(1, len(sorted_candidates) // 3)]
                selected = self._random_choice([c[0] for c in top_candidates])
                removed.append(selected)
            else:
                break
        
        new_chrom = [c for c in chromosome if c not in removed]
        return new_chrom, removed

    def complete_route_removal(self, chromosome: list[int], target_routes: int = 1, efficiency_threshold: float = 0.6) -> tuple[list[int], list[int]]:
        """
        完整路径移除算子.
        """
        if not chromosome:
            return chromosome[:], []
        
        solution = self.decoder.decode_solution(chromosome, strategy='detailed')
        
        if not solution['routes']:
            return chromosome[:], []
        
        route_efficiency = []
        for i, route_info in enumerate(solution['routes']):
            if not route_info['customers']:
                continue
            
            load_ratio = route_info['load'] / self.vehicle_capacity
            distance_per_customer = route_info['distance'] / len(route_info['customers']) if route_info['customers'] else float('inf')
            
            efficiency_score = (load_ratio * 0.7 + (1.0 / (distance_per_customer + 1)) * 0.3)
            
            route_efficiency.append({
                'route_index': i,
                'customers': route_info['customers'][:],
                'efficiency_score': efficiency_score,
            })
        
        route_efficiency.sort(key=lambda x: x['efficiency_score'])
        
        routes_to_remove_indices = []
        removed_customers = []
        
        for route_data in route_efficiency:
            if route_data['efficiency_score'] < efficiency_threshold and len(routes_to_remove_indices) < target_routes:
                routes_to_remove_indices.append(route_data['route_index'])
                removed_customers.extend(route_data['customers'])

        if not routes_to_remove_indices and route_efficiency:
             # Fallback: remove the least efficient route if none are below threshold
            selected_route = route_efficiency[0]
            routes_to_remove_indices.append(selected_route['route_index'])
            removed_customers.extend(selected_route['customers'])

        new_chromosome = [c for c in chromosome if c not in removed_customers]
        return new_chromosome, removed_customers

    def _two_opt_route(self, route: list[int], max_swaps: int = 60) -> list[int]:
        """对单条路径执行2-opt局部搜索"""
        if len(route) < 3:
            return route[:]
        r = route[:]
        n = len(r)
        improved = True
        swap_count = 0
        while improved and swap_count < max_swaps:
            improved = False
            for i in range(-1, n - 1):
                for j in range(i + 2, n):
                    a = 0 if i == -1 else r[i] + 1
                    b = r[i + 1] + 1
                    c = r[j] + 1
                    d = 0 if j == n - 1 else r[j + 1] + 1
                    
                    old = self.dist_matrix[a, b] + self.dist_matrix[c, d]
                    new = self.dist_matrix[a, c] + self.dist_matrix[b, d]
                    
                    if new + 1e-9 < old:
                        r[i + 1:j + 1] = reversed(r[i + 1:j + 1])
                        improved = True
                        swap_count += 1
                        if swap_count >= max_swaps: break
                if swap_count >= max_swaps: break
        return r

    def _get_adaptive_parameters(self, num_customers: int) -> dict:
        """
        根据客户数量动态调整局部搜索参数
        """
        def _build_common_params(two_opt, or_opt, swap, relocate, early_stop,
                                  two_opt_star, cross_exchange, three_opt,
                                  vns, als, cbls, intra_cluster, inter_cluster,
                                  num_clusters):
            return {
                'two_opt_max_iter': two_opt,
                'or_opt_max_iter': or_opt,
                'swap_max_iter': swap,
                'relocate_max_iter': relocate,
                '2opt_star_max_iter': two_opt_star,
                'cross_exchange_max_iter': cross_exchange,
                '3opt_max_iter': three_opt,
                'vns_max_iter': vns,
                'als_max_iter': als,
                'cbls_max_iter': cbls,
                'intra_cluster_iter': intra_cluster,
                'inter_cluster_iter': inter_cluster,
                'num_clusters': max(2, num_clusters),
                'early_stop_threshold': early_stop
            }

        if num_customers < 50:
            return _build_common_params(
                two_opt=40, or_opt=30, swap=30, relocate=30, early_stop=8,
                two_opt_star=25, cross_exchange=12, three_opt=15, vns=12,
                als=40, cbls=15, intra_cluster=20, inter_cluster=8,
                num_clusters=max(2, num_customers // 8)
            )
        elif num_customers < 100:
            return _build_common_params(
                two_opt=70, or_opt=55, swap=55, relocate=55, early_stop=12,
                two_opt_star=40, cross_exchange=18, three_opt=25, vns=18,
                als=55, cbls=20, intra_cluster=28, inter_cluster=12,
                num_clusters=max(3, num_customers // 10)
            )
        else:
            return _build_common_params(
                two_opt=100, or_opt=80, swap=80, relocate=80, early_stop=18,
                two_opt_star=55, cross_exchange=25, three_opt=35, vns=25,
                als=70, cbls=25, intra_cluster=36, inter_cluster=16,
                num_clusters=max(4, min(12, num_customers // 12))
            )

    def two_opt_local_search(self, chromosome: list[int], max_iterations: int = None) -> list[int]:
        """
        2-opt局部搜索：通过反转子序列来优化路径 - 字典序优化版本
        这是一个在整个染色体上操作的广义2-opt（段反转）。
        """
        current_solution = chromosome[:]
        n = len(current_solution)
        if n < 2:
            return current_solution

        params = self._get_adaptive_parameters(n)
        if max_iterations is None:
            max_iterations = params['two_opt_max_iter']
        
        early_stop_threshold = params['early_stop_threshold']
        no_improvement_count = 0
        
        for iteration in range(max_iterations):
            current_metric = self._solution_to_metric(current_solution)
            improved = False
            
            cust_to_idx = {cust: idx for idx, cust in enumerate(current_solution)}
            
            shuffled_indices = list(range(n))
            self.random_state.shuffle(shuffled_indices)
            
            for i in shuffled_indices:
                customer_i = current_solution[i]
                neighbor_customers = self._neighbor_candidates(customer_i)
                if not neighbor_customers:
                    continue

                for customer_j in neighbor_customers:
                    if customer_j not in cust_to_idx:
                        continue
                    j = cust_to_idx[customer_j]
                    
                    idx1, idx2 = sorted((i, j))
                    if idx1 == idx2:
                        continue

                    new_solution = current_solution[:idx1] + current_solution[idx1:idx2+1][::-1] + current_solution[idx2+1:]
                    new_metric = self._solution_to_metric(new_solution)
                    
                    if self._lex_better(new_metric, current_metric):
                        current_solution = new_solution
                        improved = True
                        no_improvement_count = 0
                        break  # First improvement
                if improved:
                    break
            
            if not improved:
                no_improvement_count += 1
                if no_improvement_count >= early_stop_threshold:
                    break
        
        return current_solution

    def or_opt_local_search(self, chromosome: list[int], max_iterations: int = None) -> list[int]:
        """
        Or-opt局部搜索：将长度为1, 2, 3的子序列移动到其他位置 - 字典序优化版本
        """
        current_solution = chromosome[:]
        n = len(current_solution)
        if n < 2:
            return current_solution

        params = self._get_adaptive_parameters(n)
        if max_iterations is None:
            max_iterations = params['or_opt_max_iter']
        
        early_stop_threshold = params['early_stop_threshold']
        no_improvement_count = 0
        
        for iteration in range(max_iterations):
            current_metric = self._solution_to_metric(current_solution)
            improved = False
            
            seq_lengths = [1, 2, 3]
            self.random_state.shuffle(seq_lengths)
            
            for seq_len in seq_lengths:
                if n < seq_len:
                    continue
                
                start_positions = list(range(n - seq_len + 1))
                self.random_state.shuffle(start_positions)
                
                for i in start_positions:
                    segment = current_solution[i:i+seq_len]
                    
                    temp_solution = current_solution[:i] + current_solution[i+seq_len:]
                    
                    insert_positions = list(range(len(temp_solution) + 1))
                    self.random_state.shuffle(insert_positions)
                    
                    for j in insert_positions:
                        # 避免无效移动
                        if i == j:
                            continue

                        new_solution = temp_solution[:j] + segment + temp_solution[j:]
                        new_metric = self._solution_to_metric(new_solution)
                        
                        if self._lex_better(new_metric, current_metric):
                            current_solution = new_solution
                            improved = True
                            no_improvement_count = 0
                            break
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            if not improved:
                no_improvement_count += 1
                if no_improvement_count >= early_stop_threshold:
                    break
        
        return current_solution

    def swap_local_search(self, chromosome: list[int], max_iterations: int = None) -> list[int]:
        """
        交换局部搜索：交换两个客户的位置 - 字典序优化版本
        """
        current_solution = chromosome[:]
        n = len(current_solution)
        if n < 2:
            return current_solution

        params = self._get_adaptive_parameters(n)
        if max_iterations is None:
            max_iterations = params['swap_max_iter']
        
        early_stop_threshold = params['early_stop_threshold']
        no_improvement_count = 0
        
        for iteration in range(max_iterations):
            current_metric = self._solution_to_metric(current_solution)
            improved = False
            
            indices = list(range(n))
            self.random_state.shuffle(indices)
            
            for i in range(n):
                for j in range(i + 1, n):
                    idx1, idx2 = indices[i], indices[j]
                    
                    new_solution = current_solution[:]
                    new_solution[idx1], new_solution[idx2] = new_solution[idx2], new_solution[idx1]
                    
                    new_metric = self._solution_to_metric(new_solution)
                    
                    if self._lex_better(new_metric, current_metric):
                        current_solution = new_solution
                        improved = True
                        no_improvement_count = 0
                        break
                if improved:
                    break
            
            if not improved:
                no_improvement_count += 1
                if no_improvement_count >= early_stop_threshold:
                    break
        
        return current_solution

    def relocate_local_search(self, chromosome: list[int], max_iterations: int = None) -> list[int]:
        """
        重定位局部搜索：将一个客户移动到另一个位置 - 字典序优化版本
        """
        current_solution = chromosome[:]
        n = len(current_solution)
        if n < 2:
            return current_solution

        params = self._get_adaptive_parameters(n)
        if max_iterations is None:
            max_iterations = params['relocate_max_iter']
        
        early_stop_threshold = params['early_stop_threshold']
        no_improvement_count = 0

        for iteration in range(max_iterations):
            current_metric = self._solution_to_metric(current_solution)
            improved = False
            
            indices = list(range(n))
            self.random_state.shuffle(indices)
            
            for i in indices:
                customer_to_move = current_solution[i]
                temp_solution = current_solution[:i] + current_solution[i+1:]
                
                insert_positions = list(range(len(temp_solution) + 1))
                self.random_state.shuffle(insert_positions)

                for j in insert_positions:
                    # 避免无效移动
                    if i == j:
                        continue

                    new_solution = temp_solution[:j] + [customer_to_move] + temp_solution[j:]
                    new_metric = self._solution_to_metric(new_solution)
                    
                    if self._lex_better(new_metric, current_metric):
                        current_solution = new_solution
                        improved = True
                        no_improvement_count = 0
                        break
                if improved:
                    break
            
            if not improved:
                no_improvement_count += 1
                if no_improvement_count >= early_stop_threshold:
                    break
        
        return current_solution

    def two_opt_star_local_search(self, chromosome: list[int], max_iterations: int = None) -> list[int]:
        """
        2-opt* 局部搜索：在不同路径之间交换两条边 - 字典序优化版本
        """
        current_solution = chromosome[:]
        
        try:
            routes_result = self.decoder.decode_solution(current_solution)
            if not isinstance(routes_result, dict) or 'routes' not in routes_result:
                return current_solution
            routes = routes_result['routes']
        except Exception as e:
            # 如果解码失败，返回原始解
            return current_solution
            
        if len(routes) < 2:
            return current_solution

        n = len(current_solution)
        params = self._get_adaptive_parameters(n)
        if max_iterations is None:
            max_iterations = params['2opt_star_max_iter']
        
        early_stop_threshold = params['early_stop_threshold']
        no_improvement_count = 0

        for iteration in range(max_iterations):
            current_metric = self._solution_to_metric(current_solution)
            improved = False
            
            route_indices = list(range(len(routes)))
            self.random_state.shuffle(route_indices)

            for i in range(len(routes)):
                for j in range(i + 1, len(routes)):
                    r1_idx, r2_idx = route_indices[i], route_indices[j]
                    
                    # 安全检查路径索引
                    if r1_idx >= len(routes) or r2_idx >= len(routes):
                        continue
                        
                    route1 = routes[r1_idx].get('customers', []) if isinstance(routes[r1_idx], dict) else routes[r1_idx]
                    route2 = routes[r2_idx].get('customers', []) if isinstance(routes[r2_idx], dict) else routes[r2_idx]

                    if not route1 or not route2:
                        continue

                    for c1_idx in range(len(route1)):
                        for c2_idx in range(len(route2)):
                            new_route1 = route1[:c1_idx+1] + route2[c2_idx+1:]
                            new_route2 = route2[:c2_idx+1] + route1[c1_idx+1:]

                            # 构建新的染色体
                            new_chromosome = []
                            for r_idx in range(len(routes)):
                                if r_idx == r1_idx:
                                    new_chromosome.extend(new_route1)
                                elif r_idx == r2_idx:
                                    new_chromosome.extend(new_route2)
                                else:
                                    route_customers = routes[r_idx].get('customers', []) if isinstance(routes[r_idx], dict) else routes[r_idx]
                                    new_chromosome.extend(route_customers)
                            
                            new_metric = self._solution_to_metric(new_chromosome)

                            if self._lex_better(new_metric, current_metric):
                                current_solution = new_chromosome
                                try:
                                    routes_result = self.decoder.decode_solution(current_solution)
                                    if isinstance(routes_result, dict) and 'routes' in routes_result:
                                        routes = routes_result['routes']
                                except:
                                    # 如果解码失败，继续使用当前解
                                    pass
                                improved = True
                                no_improvement_count = 0
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                no_improvement_count += 1
                if no_improvement_count >= early_stop_threshold:
                    break
        
        return current_solution

    def cross_exchange_local_search(self, chromosome: list[int], max_iterations: int = None) -> list[int]:
        """
        交叉交换局部搜索：交换两条不同路径中的客户段 - 字典序优化版本
        """
        current_solution = chromosome[:]

        def _normalize_routes(routes_data):
            """将解码结果标准化为客户索引列表的列表。"""
            normalized = []
            if not isinstance(routes_data, list):
                return normalized

            for route in routes_data:
                if isinstance(route, dict):
                    customers = None
                    for key in ("customers", "customer_sequence", "nodes"):
                        customers = route.get(key)
                        if customers is not None:
                            break
                elif isinstance(route, (list, tuple, np.ndarray)):
                    customers = list(route)
                else:
                    customers = []

                if customers is None:
                    customers = []

                normalized.append([int(c) for c in customers if isinstance(c, (int, np.integer))])

            return normalized

        def _extract_route(route_obj):
            if isinstance(route_obj, dict):
                for key in ("customers", "customer_sequence", "nodes"):
                    value = route_obj.get(key)
                    if isinstance(value, (list, tuple, np.ndarray)):
                        return [int(c) for c in value if isinstance(c, (int, np.integer))]
                return []
            if isinstance(route_obj, (list, tuple, np.ndarray)):
                return [int(c) for c in route_obj if isinstance(c, (int, np.integer))]
            return []

        try:
            decode_result = self.decoder.decode_solution(current_solution)
        except Exception:
            return current_solution

        raw_routes = []
        if isinstance(decode_result, dict):
            raw_routes = decode_result.get('routes', [])
        elif isinstance(decode_result, list):
            raw_routes = decode_result

        routes = _normalize_routes(raw_routes)
        if len(routes) < 2:
            return current_solution

        n = len(current_solution)
        params = self._get_adaptive_parameters(n)
        if max_iterations is None:
            max_iterations = params['cross_exchange_max_iter']
        max_iterations = max(1, int(max_iterations))
        early_stop_threshold = params['early_stop_threshold']
        no_improvement_count = 0

        try:
            for _ in range(max_iterations):
                current_metric = self._solution_to_metric(current_solution)
                improved = False

                route_indices = list(range(len(routes)))
                self.random_state.shuffle(route_indices)

                for i in range(len(routes)):
                    for j in range(i + 1, len(routes)):
                        r1_idx, r2_idx = route_indices[i], route_indices[j]
                        if r1_idx >= len(routes) or r2_idx >= len(routes):
                            continue

                        route1 = _extract_route(routes[r1_idx])
                        route2 = _extract_route(routes[r2_idx])
                        if not route1 or not route2:
                            continue

                        for c1_idx in range(len(route1)):
                            for c2_idx in range(len(route2)):
                                new_route1 = route1[:c1_idx + 1] + route2[c2_idx + 1:]
                                new_route2 = route2[:c2_idx + 1] + route1[c1_idx + 1:]

                                new_chromosome = []
                                for r_idx in range(len(routes)):
                                    if r_idx == r1_idx:
                                        new_chromosome.extend(new_route1)
                                    elif r_idx == r2_idx:
                                        new_chromosome.extend(new_route2)
                                    else:
                                        new_chromosome.extend(_extract_route(routes[r_idx]))

                                new_metric = self._solution_to_metric(new_chromosome)

                                if self._lex_better(new_metric, current_metric):
                                    current_solution = new_chromosome
                                    try:
                                        decode_result = self.decoder.decode_solution(current_solution)
                                    except Exception:
                                        decode_result = None

                                    if isinstance(decode_result, dict):
                                        routes = _normalize_routes(decode_result.get('routes', []))
                                    elif isinstance(decode_result, list):
                                        routes = _normalize_routes(decode_result)
                                    improved = True
                                    no_improvement_count = 0
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break

                if not improved:
                    no_improvement_count += 1
                    if no_improvement_count >= early_stop_threshold:
                        break
        except KeyError:
            return chromosome[:]

        return current_solution

    def _get_nearest_route_pairs(self, routes: list[list[int]], data: dict, k: int) -> list[tuple[int, int]]:
        """辅助函数：获取k个最近的路径对"""
        if len(routes) < 2:
            return []

        centroids = [self._calculate_route_centroid(route, data) for route in routes]
        pairs = []
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                dist = np.linalg.norm(np.array(centroids[i]) - np.array(centroids[j]))
                pairs.append(((i, j), dist))
        
        pairs.sort(key=lambda x: x[1])
        return [pair[0] for pair in pairs[:k]]

    def _calculate_route_centroid(self, route: list[int], data: dict) -> tuple[float, float]:
        """辅助函数：计算路径的质心"""
        if not route:
            return (0, 0)
        coords = [data['coords'][c] for c in route]
        return np.mean(coords, axis=0)

    def three_opt_local_search(self, chromosome: list[int], max_iterations: int = None) -> list[int]:
        """
        3-opt 局部搜索：移除三条边并以所有可能的方式重连 - 字典序优化版本
        """
        current_solution = chromosome[:]
        n = len(current_solution)
        if n < 6:
            return current_solution

        params = self._get_adaptive_parameters(n)
        if max_iterations is None:
            max_iterations = params['3opt_max_iter']
        
        early_stop_threshold = params['early_stop_threshold']
        no_improvement_count = 0

        for iteration in range(max_iterations):
            current_metric = self._solution_to_metric(current_solution)
            improved = False

            indices = list(range(n + 1))
            self.random_state.shuffle(indices)

            for i in range(n - 4):
                for j in range(i + 2, n - 2):
                    for k in range(j + 2, n):
                        # 确保i, j, k是随机选择的，以增加多样性
                        # 使用 random.sample 来避免 numpy choice 的 replace 参数问题
                        idx_i, idx_j, idx_k = random.sample(indices, 3)
                        
                        # 确保索引顺序
                        idx_i, idx_j, idx_k = sorted([idx_i, idx_j, idx_k])

                        # 避免选择相邻的边
                        if idx_j == idx_i + 1 or idx_k == idx_j + 1:
                            continue

                        # 尝试所有7种非平凡的3-opt重连方式
                        for case in range(8):
                            new_solution = self._apply_3opt_reconnection(current_solution, idx_i, idx_j, idx_k, case)
                            if new_solution is None:
                                continue

                            new_metric = self._solution_to_metric(new_solution)

                            if self._lex_better(new_metric, current_metric):
                                current_solution = new_solution
                                improved = True
                                no_improvement_count = 0
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

            if not improved:
                no_improvement_count += 1
                if no_improvement_count >= early_stop_threshold:
                    break
        
        return current_solution

    def _apply_3opt_reconnection(self, solution: list[int], i: int, j: int, k: int, case: int) -> list[int] | None:
        """辅助函数：应用3-opt重连"""
        seg1 = solution[:i+1]
        seg2 = solution[i+1:j+1]
        seg3 = solution[j+1:k+1]
        seg4 = solution[k+1:]

        if case == 0: # 2-opt (i, j)
            return seg1 + list(reversed(seg2)) + seg3 + seg4
        elif case == 1: # 2-opt (j, k)
            return seg1 + seg2 + list(reversed(seg3)) + seg4
        elif case == 2: # 2-opt (i, k)
            return seg1 + list(reversed(seg2)) + list(reversed(seg3)) + seg4
        elif case == 3:
            return seg1 + seg3 + seg2 + seg4
        elif case == 4:
            return seg1 + seg3 + list(reversed(seg2)) + seg4
        elif case == 5:
            return seg1 + list(reversed(seg3)) + seg2 + seg4
        elif case == 6:
            return seg1 + list(reversed(seg3)) + list(reversed(seg2)) + seg4
        elif case == 7: # 原始顺序
            return None
        return None

    def variable_neighborhood_search(self, chromosome: list[int], max_iterations: int = None, neighborhood_size: int = 3) -> list[int]:
        """
        可变邻域搜索 (VNS)：系统地探索不同邻域结构 - 字典序优化版本
        """
        current_solution = chromosome[:]
        n = len(current_solution)
        
        params = self._get_adaptive_parameters(n)
        if max_iterations is None:
            max_iterations = params['vns_max_iter']
        
        # 定义邻域结构列表 (函数引用)
        neighborhoods = [
            self.relocate_local_search,
            self.swap_local_search,
            self.or_opt_local_search,
            self.two_opt_local_search,
            self.two_opt_star_local_search,
            self.cross_exchange_local_search,
            self.three_opt_local_search
        ]
        
        k = 0
        iteration = 0
        while iteration < max_iterations and k < len(neighborhoods):
            current_metric = self._solution_to_metric(current_solution)
            
            # 1. 随机抖动 (在当前邻域结构中生成一个随机解)
            #    这里我们简化为直接应用邻域搜索算子
            
            # 2. 局部搜索
            #    从第k个邻域开始搜索
            search_operator = neighborhoods[k]
            # VNS通常需要多次迭代或更复杂的邻域切换逻辑，这里简化为单次调用
            new_solution = search_operator(current_solution, max_iterations=params.get(f"{search_operator.__name__}_max_iter", 20))
            
            new_metric = self._solution_to_metric(new_solution)
            
            # 3. 移动或不移动
            if self._lex_better(new_metric, current_metric):
                current_solution = new_solution
                k = 0  # 如果找到更好的解，回到第一个邻域
            else:
                k += 1 # 否则，探索下一个邻域
            
            iteration += 1
            
        return current_solution

    def adaptive_local_search(self, chromosome: list[int], max_iterations: int = None, initial_ro: float = 0.1) -> list[int]:
        """
        自适应局部搜索：根据算子性能动态选择局部搜索策略 - 字典序优化版本
        """
        current_solution = chromosome[:]
        n = len(current_solution)

        params = self._get_adaptive_parameters(n)
        if max_iterations is None:
            max_iterations = params['als_max_iter']

        operators = [
            self.two_opt_local_search,
            self.or_opt_local_search,
            self.swap_local_search,
            self.relocate_local_search,
            self.two_opt_star_local_search,
            self.cross_exchange_local_search,
            self.three_opt_local_search
        ]

        # 初始化算子性能统计
        if not hasattr(self, 'operator_stats'):
            self.operator_stats = {op.__name__: {'score': 1, 'num_applications': 1} for op in operators}

        ro = initial_ro

        for iteration in range(max_iterations):
            current_metric = self._solution_to_metric(current_solution)

            # --- 算子选择 ---
            total_score = sum(stat['score'] for stat in self.operator_stats.values())
            probabilities = [self.operator_stats[op.__name__]['score'] / total_score for op in operators]
            selected_op = self._random_choice(operators, probabilities)

            # --- 应用算子 ---
            new_solution = selected_op(current_solution, max_iterations=10) # 运行少量迭代
            new_metric = self._solution_to_metric(new_solution)

            # --- 性能更新 ---
            op_name = selected_op.__name__
            self.operator_stats[op_name]['num_applications'] += 1
            
            improvement = 0
            if self._lex_better(new_metric, current_metric):
                improvement = 1 # 简单地将改进设为1
                current_solution = new_solution
            
            # 更新分数：滑动平均
            old_score = self.operator_stats[op_name]['score']
            num_apps = self.operator_stats[op_name]['num_applications']
            self.operator_stats[op_name]['score'] = (1 - ro) * old_score + ro * improvement

            # 动态调整ro (学习率)
            if iteration % 50 == 0 and iteration > 0:
                ro = max(0.01, ro * 0.95)

        return current_solution

    def run_alns(self, initial_chromosome: list[int], iterations: int = 1000, remove_frac: float = 0.2, temperature: float = 1000, cooling: float = 0.998, verbose: bool = False):
        """
        执行ALNS算法主循环.
        """
        # 算子和它们的权重
        destroy_operators = {
            self.random_removal: 1,
            self.shaw_removal: 3,
            self.complete_route_removal: 2,
        }
        repair_operators = {
            self.greedy_repair: 2,
            self.regret_insertion: 3,
        }

        # 优化的自适应权重调整参数
        reward_good = 2      # 降低一般改进的奖励
        reward_better = 4    # 适中的较好改进奖励
        reward_best = 8      # 降低最佳改进奖励，避免过度偏向某个算子
        decay = 0.95         # 更慢的衰减，保持算子权重的稳定性

        current_chromosome = initial_chromosome[:]
        best_chromosome = initial_chromosome[:]
        current_cost = self._solution_to_metric(current_chromosome)[1]
        best_cost = current_cost

        for i in range(iterations):
            # 选择破坏和修复算子
            destroy_op = random.choices(list(destroy_operators.keys()), weights=list(destroy_operators.values()))[0]
            repair_op = random.choices(list(repair_operators.keys()), weights=list(repair_operators.values()))[0]

            # 应用算子
            destroyed_chrom, removed = destroy_op(current_chromosome, remove_count=int(len(current_chromosome) * remove_frac))
            new_chromosome = repair_op(destroyed_chrom, removed)
            
            new_cost_metric = self._solution_to_metric(new_chromosome)
            new_cost = new_cost_metric[1]

            # 模拟退火接受准则
            accepted = False
            if self._lex_better(new_cost_metric, self._solution_to_metric(current_chromosome)):
                accepted = True
                # 更新权重
                destroy_operators[destroy_op] = destroy_operators[destroy_op] * decay + reward_better
                repair_operators[repair_op] = repair_operators[repair_op] * decay + reward_better
            elif new_cost < current_cost:
                 accepted = True
                 destroy_operators[destroy_op] = destroy_operators[destroy_op] * decay + reward_good
                 repair_operators[repair_op] = repair_operators[repair_op] * decay + reward_good
            else:
                delta = new_cost - current_cost
                if random.random() < np.exp(-delta / temperature):
                    accepted = True
            
            if accepted:
                current_chromosome = new_chromosome
                current_cost = new_cost
                if self._lex_better(new_cost_metric, self._solution_to_metric(best_chromosome)):
                    best_chromosome = new_chromosome
                    best_cost = new_cost
                    # 发现全局最优解，给予最高奖励
                    destroy_operators[destroy_op] = destroy_operators[destroy_op] * decay + reward_best
                    repair_operators[repair_op] = repair_operators[repair_op] * decay + reward_best

            # 降温
            temperature *= cooling

            if verbose and i % 100 == 0:
                print(f"Iteration {i}: Best Cost = {best_cost:.2f}, Current Cost = {current_cost:.2f}, Temp = {temperature:.2f}")

        return best_chromosome, best_cost

    def lin_kernighan_local_search(self, chromosome: list[int], max_iterations: int = None) -> list[int]:
        """Lin-Kernighan启发式局部搜索"""
        if max_iterations is None:
            max_iterations = min(50, len(chromosome))
        
        current = chromosome[:]
        current_cost = self._solution_to_metric(current)[1]
        best = current[:]
        best_cost = current_cost
        
        for _ in range(max_iterations):
            improved = False
            
            # 尝试k-opt改进，k从2到4
            for k in range(2, min(5, len(current))):
                if self._k_opt_improve(current, k):
                    new_cost = self._solution_to_metric(current)[1]
                    if new_cost < current_cost:
                        current_cost = new_cost
                        improved = True
                        if new_cost < best_cost:
                            best = current[:]
                            best_cost = new_cost
                        break
            
            if not improved:
                break
        
        return best

    def _k_opt_improve(self, chromosome: list[int], k: int) -> bool:
        """执行k-opt改进"""
        n = len(chromosome)
        if n < k + 1:
            return False
        
        # 随机选择k个边进行重连
        edges = random.sample(range(n), k)
        edges.sort()
        
        # 尝试重连
        segments = []
        start = 0
        for edge in edges:
            segments.append(chromosome[start:edge+1])
            start = edge + 1
        if start < n:
            segments.append(chromosome[start:])
        
        # 随机重排序列
        random.shuffle(segments)
        new_chromosome = []
        for segment in segments:
            if random.random() < 0.5:
                segment.reverse()
            new_chromosome.extend(segment)
        
        chromosome[:] = new_chromosome
        return True

    def large_neighborhood_search(self, chromosome: list[int], max_iterations: int = None) -> list[int]:
        """大邻域搜索"""
        if max_iterations is None:
            max_iterations = min(30, len(chromosome) // 2)
        
        current = chromosome[:]
        best = current[:]
        best_cost = self._solution_to_metric(best)[1]
        
        for iteration in range(max_iterations):
            # 大规模破坏和重构
            remove_count = max(3, len(current) // 4)  # 移除25%的客户
            
            # 使用多种破坏策略
            if iteration % 3 == 0:
                destroyed, removed = self.shaw_removal(current, remove_count)
            elif iteration % 3 == 1:
                destroyed, removed = self.random_removal(current, remove_count)
            else:
                destroyed, removed = self.complete_route_removal(current, 1)
            
            # 使用多种修复策略
            if iteration % 2 == 0:
                repaired = self.greedy_repair(destroyed, removed)
            else:
                repaired = self.regret_insertion(destroyed, removed)
            
            # 局部优化
            repaired = self.two_opt_local_search(repaired, 10)
            
            new_cost = self._solution_to_metric(repaired)[1]
            
            # 改进的模拟退火温度调度 - 使用指数衰减和线性衰减的组合
            initial_temp = 100
            final_temp = 0.1
            # 使用更平滑的温度衰减策略
            temperature = max(final_temp, initial_temp * (0.98 ** iteration))
            if new_cost < best_cost or random.random() < math.exp(-(new_cost - best_cost) / temperature):
                current = repaired[:]
                if new_cost < best_cost:
                    best = repaired[:]
                    best_cost = new_cost
        
        return best

    def path_relinking_local_search(self, chromosome1: list[int], chromosome2: list[int]) -> list[int]:
        """路径重连局部搜索"""
        if len(chromosome1) != len(chromosome2):
            return chromosome1[:]
        
        best = chromosome1[:]
        best_cost = self._solution_to_metric(best)[1]
        
        current = chromosome1[:]
        target = chromosome2[:]
        
        # 逐步将current转换为target
        while current != target:
            # 找到第一个不同的位置
            diff_pos = -1
            for i in range(len(current)):
                if current[i] != target[i]:
                    diff_pos = i
                    break
            
            if diff_pos == -1:
                break
            
            # 找到target[diff_pos]在current中的位置
            target_val = target[diff_pos]
            current_pos = current.index(target_val)
            
            # 交换位置
            current[diff_pos], current[current_pos] = current[current_pos], current[diff_pos]
            
            # 评估新解
            new_cost = self._solution_to_metric(current)[1]
            if new_cost < best_cost:
                best = current[:]
                best_cost = new_cost
        
        return best

    def guided_local_search(self, chromosome: list[int], max_iterations: int = None) -> list[int]:
        """引导局部搜索"""
        if max_iterations is None:
            max_iterations = min(100, len(chromosome) * 2)
        
        current = chromosome[:]
        best = current[:]
        best_cost = self._solution_to_metric(best)[1]
        
        # 特征惩罚权重
        penalties = defaultdict(float)
        lambda_param = 0.1
        
        for iteration in range(max_iterations):
            # 计算增强成本（原成本 + 惩罚）
            augmented_cost = self._calculate_augmented_cost(current, penalties, lambda_param)
            
            # 执行局部搜索
            improved = current[:]
            improved = self.two_opt_local_search(improved, 5)
            improved = self.or_opt_local_search(improved, 5)
            
            new_cost = self._solution_to_metric(improved)[1]
            new_augmented_cost = self._calculate_augmented_cost(improved, penalties, lambda_param)
            
            # 更新当前解
            if new_augmented_cost < augmented_cost:
                current = improved[:]
                if new_cost < best_cost:
                    best = improved[:]
                    best_cost = new_cost
            else:
                # 增加特征惩罚
                features = self._extract_features(current)
                for feature in features:
                    penalties[feature] += 1.0
        
        return best

    def _calculate_augmented_cost(self, chromosome: list[int], penalties: dict, lambda_param: float) -> float:
        """计算增强成本"""
        base_cost = self._solution_to_metric(chromosome)[1]
        penalty_cost = 0.0
        
        features = self._extract_features(chromosome)
        for feature in features:
            penalty_cost += penalties.get(feature, 0.0)
        
        return base_cost + lambda_param * penalty_cost

    def _extract_features(self, chromosome: list[int]) -> list[str]:
        """提取解的特征"""
        features = []
        
        # 边特征
        for i in range(len(chromosome) - 1):
            edge = f"edge_{min(chromosome[i], chromosome[i+1])}_{max(chromosome[i], chromosome[i+1])}"
            features.append(edge)
        
        # 三元组特征
        for i in range(len(chromosome) - 2):
            triplet = f"triplet_{chromosome[i]}_{chromosome[i+1]}_{chromosome[i+2]}"
            features.append(triplet)
        
        return features

    def cluster_based_local_search(self, chromosome: list[int], max_iterations: int = None) -> list[int]:
        """
        基于聚类的局部搜索：将客户分组并分别进行簇内和簇间优化 - 字典序优化版本
        """
        current_solution = chromosome[:]
        n = len(current_solution)
    
        params = self._get_adaptive_parameters(n)
        if max_iterations is None:
            max_iterations = params['cbls_max_iter']
    
        for iteration in range(max_iterations):
            current_metric = self._solution_to_metric(current_solution)
            
            # 1. 创建客户聚类
            clusters = self._create_customer_clusters(current_solution, self.data, n_clusters=params['num_clusters'])
            
            # 2. 簇内优化
            optimized_clusters = self._perform_intra_cluster_optimization(clusters, params)
            
            # 3. 簇间优化
            new_clusters = self._perform_inter_cluster_optimization(optimized_clusters, params)
            
            # 4. 重构解
            new_solution_list = []
            for cluster in new_clusters:
                new_solution_list.extend(cluster['customers'])
            
            # 确保所有客户都被包含
            if len(set(new_solution_list)) == n:
                new_metric = self._solution_to_metric(new_solution_list)
                if self._lex_better(new_metric, current_metric):
                    current_solution = new_solution_list
            
        return current_solution
    
    def _create_customer_clusters(self, chromosome: list[int], data: dict, n_clusters: int) -> list[dict]:
        """辅助函数：创建客户聚类"""
        coords = np.array([data['coords'][c] for c in chromosome])
        demands = np.array([data['demands'][c] for c in chromosome])
        time_windows = np.array([data['time_windows'][c] for c in chromosome])
        
        try:
            # 尝试使用KMeans进行地理位置聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(coords)
        except Exception:
            # 如果KMeans失败，则回退到基于容量的简单聚类
            return self._fallback_capacity_clustering(chromosome, data)
    
        clusters = {i: {'customers': [], 'total_demand': 0, 'centroid': list(kmeans.cluster_centers_[i])} for i in range(n_clusters)}
        
        for i, cust_idx in enumerate(chromosome):
            label = labels[i]
            clusters[label]['customers'].append(cust_idx)
            clusters[label]['total_demand'] += demands[i]
    
        # 调整聚类以满足容量约束
        clusters = self._adjust_clusters_for_capacity(list(clusters.values()), data['vehicle_capacity'], data)
        return clusters
    
    def _fallback_capacity_clustering(self, chromosome: list[int], data: dict) -> list[dict]:
        """辅助函数：当KMeans失败时的回退聚类方法"""
        clusters = []
        current_cluster = {'customers': [], 'total_demand': 0}
        for cust_idx in chromosome:
            demand = data['demands'][cust_idx]
            if current_cluster['total_demand'] + demand > data['vehicle_capacity'] and current_cluster['customers']:
                clusters.append(current_cluster)
                current_cluster = {'customers': [], 'total_demand': 0}
            current_cluster['customers'].append(cust_idx)
            current_cluster['total_demand'] += demand
        if current_cluster['customers']:
            clusters.append(current_cluster)
        return clusters
    
    def _adjust_clusters_for_capacity(self, clusters: list[dict], capacity: int, data: dict) -> list[dict]:
        """辅助函数：调整聚类以满足车辆容量约束"""
        adjusted = False
        for i, cluster in enumerate(clusters):
            if cluster['total_demand'] > capacity:
                adjusted = True
                new_clusters = self._intelligent_cluster_split(cluster, capacity, data)
                clusters.pop(i)
                clusters.extend(new_clusters)
        
        return clusters if not adjusted else self._adjust_clusters_for_capacity(clusters, capacity, data)
    
    def _intelligent_cluster_split(self, cluster: dict, capacity: int, data: dict) -> list[dict]:
        """辅助函数：智能地拆分超载的聚类"""
        # 按时间窗和地理位置排序客户
        customers = sorted(cluster['customers'], key=lambda c: (data['time_windows'][c][0], data['coords'][c][0], data['coords'][c][1]))
        
        new_clusters = []
        current_cluster = {'customers': [], 'total_demand': 0}
        for cust_idx in customers:
            demand = data['demands'][cust_idx]
            if current_cluster['total_demand'] + demand > capacity and current_cluster['customers']:
                new_clusters.append(current_cluster)
                current_cluster = {'customers': [], 'total_demand': 0}
            current_cluster['customers'].append(cust_idx)
            current_cluster['total_demand'] += demand
        if current_cluster['customers']:
            new_clusters.append(current_cluster)
        return new_clusters
    
    def _perform_intra_cluster_optimization(self, clusters: list[dict], params: dict) -> list[dict]:
        """辅助函数：在每个聚类内部进行局部搜索优化"""
        optimized_clusters = []
        for cluster in clusters:
            if len(cluster['customers']) > 1:
                # 对每个簇内的客户应用局部搜索
                optimized_customers = self.two_opt_local_search(cluster['customers'], max_iterations=params['intra_cluster_iter'])
                cluster['customers'] = optimized_customers
            optimized_clusters.append(cluster)
        return optimized_clusters
    
    def _perform_inter_cluster_optimization(self, clusters: list[dict], params: dict) -> list[dict]:
        """辅助函数：在聚类之间进行优化（例如，客户交换）"""
        for _ in range(params['inter_cluster_iter']):
            if len(clusters) < 2:
                break

            # 获取相邻的聚类对
            adjacent_pairs = self.get_adjacent_clusters(clusters, k=3)
            if not adjacent_pairs:
                break

            for idx1, idx2 in adjacent_pairs:
                cluster1, cluster2 = clusters[idx1], clusters[idx2]

                # 尝试在两个聚类之间移动客户
                for cust_idx_c1 in list(cluster1['customers']):
                    demand_c1 = self.data['demands'][cust_idx_c1]
                    if cluster2['total_demand'] + demand_c1 <= self.data['vehicle_capacity']:
                        # 评估移动带来的成本变化
                        # 这是一个简化的评估，实际应用中需要更精确的成本计算
                        original_dist = self._route_distance([cust_idx_c1], self.data)
                        new_dist = self._route_distance(cluster2['customers'] + [cust_idx_c1], self.data)
                        
                        if new_dist < original_dist: # 简化决策
                            cluster1['customers'].remove(cust_idx_c1)
                            cluster1['total_demand'] -= demand_c1
                            cluster2['customers'].append(cust_idx_c1)
                            cluster2['total_demand'] += demand_c1
                            break # 移动一个后重新评估

        return clusters

    def get_adjacent_clusters(self, clusters: list[dict], k: int) -> list[tuple[int, int]]:
        """辅助函数：获取k个最近的聚类对"""
        if len(clusters) < 2:
            return []
        
        # 更新质心
        for cluster in clusters:
            if cluster['customers']:
                cluster['centroid'] = self._calculate_route_centroid(cluster['customers'], self.data)
            else:
                # 如果聚类为空，则不能计算质心
                cluster['centroid'] = (0,0) # 或者其他默认值

        pairs = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if not clusters[i]['customers'] or not clusters[j]['customers']:
                    continue
                dist = np.linalg.norm(np.array(clusters[i]['centroid']) - np.array(clusters[j]['centroid']))
                pairs.append(((i, j), dist))
        
        pairs.sort(key=lambda x: x[1])
        return [pair[0] for pair in pairs[:k]]

    def run_alns(self, initial_chromosome: list[int], iterations: int = 1000, remove_frac: float = 0.2, temperature: float = 1000, cooling: float = 0.998, verbose: bool = False):
        """
        执行ALNS算法主循环.
        """
        # 算子和它们的权重
        destroy_operators = {
            self.random_removal: 1,
            self.shaw_removal: 3,
            self.complete_route_removal: 2,
        }
        repair_operators = {
            self.greedy_repair: 2,
            self.regret_insertion: 3,
        }

        # 自适应权重调整参数
        reward_good = 3
        reward_better = 5
        reward_best = 10
        decay = 0.9

        current_chromosome = initial_chromosome[:]
        best_chromosome = initial_chromosome[:]
        current_cost = self._solution_to_metric(current_chromosome)[1]
        best_cost = current_cost

        for i in range(iterations):
            # 选择破坏和修复算子
            destroy_op = random.choices(list(destroy_operators.keys()), weights=list(destroy_operators.values()))[0]
            repair_op = random.choices(list(repair_operators.keys()), weights=list(repair_operators.values()))[0]

            # 应用算子
            destroyed_chrom, removed = destroy_op(current_chromosome, remove_count=int(len(current_chromosome) * remove_frac))
            new_chromosome = repair_op(destroyed_chrom, removed)
            
            new_cost_metric = self._solution_to_metric(new_chromosome)
            new_cost = new_cost_metric[1]

            # 模拟退火接受准则
            accepted = False
            if self._lex_better(new_cost_metric, self._solution_to_metric(current_chromosome)):
                accepted = True
                # 更新权重
                destroy_operators[destroy_op] = destroy_operators[destroy_op] * decay + reward_better
                repair_operators[repair_op] = repair_operators[repair_op] * decay + reward_better
            elif new_cost < current_cost:
                 accepted = True
                 destroy_operators[destroy_op] = destroy_operators[destroy_op] * decay + reward_good
                 repair_operators[repair_op] = repair_operators[repair_op] * decay + reward_good
            else:
                delta = new_cost - current_cost
                if random.random() < np.exp(-delta / temperature):
                    accepted = True
            
            if accepted:
                current_chromosome = new_chromosome
                current_cost = new_cost
                if self._lex_better(new_cost_metric, self._solution_to_metric(best_chromosome)):
                    best_chromosome = new_chromosome
                    best_cost = new_cost
                    # 发现全局最优解，给予最高奖励
                    destroy_operators[destroy_op] = destroy_operators[destroy_op] * decay + reward_best
                    repair_operators[repair_op] = repair_operators[repair_op] * decay + reward_best

            # 降温
            temperature *= cooling

            if verbose and i % 100 == 0:
                print(f"Iteration {i}: Best Cost = {best_cost:.2f}, Current Cost = {current_cost:.2f}, Temp = {temperature:.2f}")

        return best_chromosome, best_cost