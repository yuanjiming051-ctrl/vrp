# File: worst_route_optimizer.py
# 最差路径局部搜索算法 - 针对最差路径的专门优化策略

import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from decoder import VRPTWDecoder


class WorstRouteOptimizer:
    """
    最差路径局部搜索优化器
    
    该算法在每次迭代操作后，对当前解的最差路径执行局部优化操作，
    包含路径评估机制、最差路径识别、局部操作策略和迭代更新规则。
    """
    
    def __init__(self, instance, decoder: VRPTWDecoder, verbose: bool = True):
        """
        初始化最差路径优化器
        
        Args:
            instance: VRPTW实例
            decoder: VRPTW解码器
            verbose: 是否输出详细信息
        """
        self.instance = instance
        self.decoder = decoder
        self.verbose = verbose
        self.distance_matrix = instance.distance_matrix
        
        # 路径评估权重参数
        self.distance_weight = 0.4      # 距离成本权重
        self.time_penalty_weight = 0.3  # 时间窗违反惩罚权重
        self.capacity_penalty_weight = 0.2  # 容量违反惩罚权重
        self.efficiency_weight = 0.1    # 路径效率权重（距离/客户数）
        
    def evaluate_route_quality(self, route: List[int]) -> Dict[str, float]:
        """
        1. 路径评估机制：评估单条路径的质量
        
        评估指标包括：
        - 距离成本：路径的总距离
        - 时间窗违反：违反时间窗的惩罚
        - 容量违反：违反容量约束的惩罚
        - 路径效率：距离与客户数的比值
        - 综合质量分数：加权综合评分
        
        Args:
            route: 路径中的客户列表
            
        Returns:
            包含各项评估指标的字典
        """
        if not route:
            return {
                'distance_cost': 0.0,
                'time_penalty': 0.0,
                'capacity_penalty': 0.0,
                'efficiency_score': 0.0,
                'quality_score': 0.0
            }
        
        # 计算距离成本
        distance_cost = self._calculate_route_distance(route)
        
        # 计算时间窗违反惩罚
        time_penalty = self._calculate_time_penalty(route)
        
        # 计算容量违反惩罚
        capacity_penalty = self._calculate_capacity_penalty(route)
        
        # 计算路径效率（距离/客户数，越小越好）
        efficiency_score = distance_cost / len(route) if len(route) > 0 else float('inf')
        
        # 计算综合质量分数（越大表示路径质量越差）
        quality_score = (
            self.distance_weight * distance_cost +
            self.time_penalty_weight * time_penalty +
            self.capacity_penalty_weight * capacity_penalty +
            self.efficiency_weight * efficiency_score
        )
        
        return {
            'distance_cost': distance_cost,
            'time_penalty': time_penalty,
            'capacity_penalty': capacity_penalty,
            'efficiency_score': efficiency_score,
            'quality_score': quality_score
        }
    
    def identify_worst_route(self, chromosome: List[int]) -> Tuple[int, List[int], Dict[str, float]]:
        """
        2. 最差路径识别：准确识别当前解中的最差路径
        
        Args:
            chromosome: 完整的染色体解
            
        Returns:
            (worst_route_index, worst_route, worst_route_metrics)
        """
        # 解码染色体获取所有路径
        solution = self.decoder.decode_solution(chromosome, strategy='detailed')
        routes = solution['routes']
        
        if not routes:
            return -1, [], {}
        
        worst_route_idx = -1
        worst_route = []
        worst_quality_score = -1.0
        worst_metrics = {}
        
        # 评估每条路径的质量
        for i, route_info in enumerate(routes):
            route_customers = route_info['customers']
            metrics = self.evaluate_route_quality(route_customers)
            
            if self.verbose:
                print(f"    路径 {i}: 客户 {route_customers}, 质量分数: {metrics['quality_score']:.2f}")
            
            # 找到质量分数最高（最差）的路径
            if metrics['quality_score'] > worst_quality_score:
                worst_quality_score = metrics['quality_score']
                worst_route_idx = i
                worst_route = route_customers.copy()
                worst_metrics = metrics.copy()
        
        if self.verbose:
            print(f"  识别最差路径: 路径 {worst_route_idx}, 客户 {worst_route}, 质量分数: {worst_quality_score:.2f}")
        
        return worst_route_idx, worst_route, worst_metrics
    
    def optimize_worst_route(self, chromosome: List[int], max_iterations: int = 20) -> Tuple[List[int], bool]:
        """
        3. 局部操作策略：定义针对最差路径的具体优化方法
        
        优化策略包括：
        - 2-opt局部搜索
        - Or-opt重定位
        - 客户交换
        - 路径重构
        
        Args:
            chromosome: 当前染色体解
            max_iterations: 最大迭代次数
            
        Returns:
            (optimized_chromosome, improved)
        """
        if self.verbose:
            print(f"  [最差路径优化] 开始优化，最大迭代次数: {max_iterations}")
        
        current_chromosome = chromosome.copy()
        best_chromosome = chromosome.copy()
        best_cost = self._evaluate_chromosome(chromosome)
        improved = False
        
        for iteration in range(max_iterations):
            # 识别当前最差路径
            worst_idx, worst_route, worst_metrics = self.identify_worst_route(current_chromosome)
            
            if worst_idx == -1 or len(worst_route) <= 1:
                break
            
            # 应用多种局部优化策略
            strategies = [
                self._apply_2opt_to_route,
                self._apply_or_opt_to_route,
                self._apply_customer_swap,
                self._apply_route_reconstruction
            ]
            
            iteration_improved = False
            for strategy in strategies:
                try:
                    optimized_chromosome = strategy(current_chromosome, worst_idx, worst_route)
                    optimized_cost = self._evaluate_chromosome(optimized_chromosome)
                    
                    if optimized_cost < best_cost:
                        best_chromosome = optimized_chromosome.copy()
                        best_cost = optimized_cost
                        current_chromosome = optimized_chromosome.copy()
                        improved = True
                        iteration_improved = True
                        
                        if self.verbose:
                            print(f"    迭代 {iteration+1}: {strategy.__name__} 改进成本 {optimized_cost:.2f}")
                        break
                        
                except Exception as e:
                    if self.verbose:
                        print(f"    策略 {strategy.__name__} 失败: {str(e)}")
                    continue
            
            # 如果本次迭代没有改进，尝试随机扰动
            if not iteration_improved:
                current_chromosome = self._apply_random_perturbation(current_chromosome, worst_idx)
        
        return best_chromosome, improved
    
    def iterative_worst_route_optimization(self, chromosome: List[int], 
                                         max_outer_iterations: int = 5,
                                         max_inner_iterations: int = 20) -> List[int]:
        """
        4. 迭代更新规则：确保每次优化后及时更新解决方案
        
        Args:
            chromosome: 初始染色体解
            max_outer_iterations: 最大外层迭代次数
            max_inner_iterations: 每次最差路径优化的最大迭代次数
            
        Returns:
            优化后的染色体解
        """
        if self.verbose:
            print(f"[迭代最差路径优化] 开始，外层迭代: {max_outer_iterations}, 内层迭代: {max_inner_iterations}")
        
        current_chromosome = chromosome.copy()
        initial_cost = self._evaluate_chromosome(current_chromosome)
        
        if self.verbose:
            print(f"  初始成本: {initial_cost:.2f}")
        
        for outer_iter in range(max_outer_iterations):
            if self.verbose:
                print(f"\n  外层迭代 {outer_iter + 1}/{max_outer_iterations}")
            
            # 优化当前最差路径
            optimized_chromosome, improved = self.optimize_worst_route(
                current_chromosome, max_inner_iterations
            )
            
            if improved:
                current_chromosome = optimized_chromosome
                current_cost = self._evaluate_chromosome(current_chromosome)
                
                if self.verbose:
                    improvement = initial_cost - current_cost
                    print(f"    外层迭代 {outer_iter + 1} 改进: {current_cost:.2f} (总改进: {improvement:.2f})")
            else:
                if self.verbose:
                    print(f"    外层迭代 {outer_iter + 1} 无改进，停止优化")
                break
        
        final_cost = self._evaluate_chromosome(current_chromosome)
        total_improvement = initial_cost - final_cost
        
        if self.verbose:
            print(f"\n[迭代最差路径优化] 完成，最终成本: {final_cost:.2f}, 总改进: {total_improvement:.2f}")
        
        return current_chromosome
    
    # ==================== 辅助方法 ====================
    
    def _calculate_route_distance(self, route: List[int]) -> float:
        """计算路径的总距离"""
        if not route:
            return 0.0
        
        distance = 0.0
        # depot → first customer
        distance += self.distance_matrix[0, route[0] + 1]
        
        # between customers
        for i in range(len(route) - 1):
            distance += self.distance_matrix[route[i] + 1, route[i + 1] + 1]
        
        # last customer → depot
        distance += self.distance_matrix[route[-1] + 1, 0]
        
        return distance
    
    def _calculate_time_penalty(self, route: List[int]) -> float:
        """计算时间窗违反惩罚"""
        if not route:
            return 0.0
        
        penalty = 0.0
        current_time = 0.0
        current_pos = 0  # depot
        
        for customer_idx in route:
            customer = self.instance.ordinary_customers[customer_idx]
            
            # 计算到达时间
            travel_time = self.distance_matrix[current_pos, customer_idx + 1]
            arrival_time = current_time + travel_time
            
            # 检查时间窗违反
            if arrival_time > customer['due_date']:
                penalty += (arrival_time - customer['due_date']) * 10  # 惩罚系数
            
            # 更新当前时间和位置
            service_start = max(arrival_time, customer['ready_time'])
            current_time = service_start + customer['service_time']
            current_pos = customer_idx + 1
        
        return penalty
    
    def _calculate_capacity_penalty(self, route: List[int]) -> float:
        """计算容量违反惩罚"""
        if not route:
            return 0.0
        
        total_demand = sum(self.instance.ordinary_customers[i]['demand'] for i in route)
        capacity_violation = max(0, total_demand - self.instance.vehicle_capacity)
        
        return capacity_violation * 20  # 惩罚系数
    
    def _evaluate_chromosome(self, chromosome: List[int]) -> float:
        """评估染色体的总成本"""
        solution = self.decoder.decode_solution(chromosome, strategy='fast')
        return solution['total_distance']
    
    def _apply_2opt_to_route(self, chromosome: List[int], route_idx: int, route: List[int]) -> List[int]:
        """对指定路径应用2-opt优化"""
        if len(route) < 3:
            return chromosome.copy()
        
        # 找到路径在染色体中的位置
        route_positions = self._find_route_positions_in_chromosome(chromosome, route)
        if not route_positions:
            return chromosome.copy()
        
        best_chromosome = chromosome.copy()
        best_cost = self._evaluate_chromosome(chromosome)
        
        # 尝试所有可能的2-opt交换
        for i in range(len(route)):
            for j in range(i + 2, len(route)):
                # 创建新的路径
                new_route = route.copy()
                new_route[i:j+1] = new_route[i:j+1][::-1]
                
                # 更新染色体
                new_chromosome = chromosome.copy()
                for k, pos in enumerate(route_positions):
                    new_chromosome[pos] = new_route[k]
                
                # 评估新解
                new_cost = self._evaluate_chromosome(new_chromosome)
                if new_cost < best_cost:
                    best_chromosome = new_chromosome
                    best_cost = new_cost
        
        return best_chromosome
    
    def _apply_or_opt_to_route(self, chromosome: List[int], route_idx: int, route: List[int]) -> List[int]:
        """对指定路径应用Or-opt优化（重定位）"""
        if len(route) < 2:
            return chromosome.copy()
        
        route_positions = self._find_route_positions_in_chromosome(chromosome, route)
        if not route_positions:
            return chromosome.copy()
        
        best_chromosome = chromosome.copy()
        best_cost = self._evaluate_chromosome(chromosome)
        
        # 尝试移动不同长度的子序列
        for seq_len in [1, 2, 3]:
            if len(route) < seq_len:
                continue
                
            for i in range(len(route) - seq_len + 1):
                segment = route[i:i+seq_len]
                remaining = route[:i] + route[i+seq_len:]
                
                # 尝试插入到不同位置
                for j in range(len(remaining) + 1):
                    new_route = remaining[:j] + segment + remaining[j:]
                    
                    # 更新染色体
                    new_chromosome = chromosome.copy()
                    for k, pos in enumerate(route_positions):
                        new_chromosome[pos] = new_route[k]
                    
                    # 评估新解
                    new_cost = self._evaluate_chromosome(new_chromosome)
                    if new_cost < best_cost:
                        best_chromosome = new_chromosome
                        best_cost = new_cost
        
        return best_chromosome
    
    def _apply_customer_swap(self, chromosome: List[int], route_idx: int, route: List[int]) -> List[int]:
        """在路径内部交换客户位置"""
        if len(route) < 2:
            return chromosome.copy()
        
        route_positions = self._find_route_positions_in_chromosome(chromosome, route)
        if not route_positions:
            return chromosome.copy()
        
        best_chromosome = chromosome.copy()
        best_cost = self._evaluate_chromosome(chromosome)
        
        # 尝试交换路径中的任意两个客户
        for i in range(len(route)):
            for j in range(i + 1, len(route)):
                new_route = route.copy()
                new_route[i], new_route[j] = new_route[j], new_route[i]
                
                # 更新染色体
                new_chromosome = chromosome.copy()
                for k, pos in enumerate(route_positions):
                    new_chromosome[pos] = new_route[k]
                
                # 评估新解
                new_cost = self._evaluate_chromosome(new_chromosome)
                if new_cost < best_cost:
                    best_chromosome = new_chromosome
                    best_cost = new_cost
        
        return best_chromosome
    
    def _apply_route_reconstruction(self, chromosome: List[int], route_idx: int, route: List[int]) -> List[int]:
        """重构路径（随机重排）"""
        if len(route) < 2:
            return chromosome.copy()
        
        route_positions = self._find_route_positions_in_chromosome(chromosome, route)
        if not route_positions:
            return chromosome.copy()
        
        # 随机重排路径中的客户
        new_route = route.copy()
        random.shuffle(new_route)
        
        # 更新染色体
        new_chromosome = chromosome.copy()
        for k, pos in enumerate(route_positions):
            new_chromosome[pos] = new_route[k]
        
        return new_chromosome
    
    def _apply_random_perturbation(self, chromosome: List[int], route_idx: int) -> List[int]:
        """对染色体应用随机扰动"""
        new_chromosome = chromosome.copy()
        
        # 随机交换两个客户
        if len(new_chromosome) >= 2:
            i, j = random.sample(range(len(new_chromosome)), 2)
            new_chromosome[i], new_chromosome[j] = new_chromosome[j], new_chromosome[i]
        
        return new_chromosome
    
    def _find_route_positions_in_chromosome(self, chromosome: List[int], route: List[int]) -> List[int]:
        """找到路径中客户在染色体中的位置"""
        positions = []
        route_set = set(route)
        
        for i, customer in enumerate(chromosome):
            if customer in route_set:
                positions.append(i)
                route_set.remove(customer)
                if not route_set:
                    break
        
        return positions