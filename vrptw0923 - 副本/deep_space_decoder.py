#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的Deep空间解码器
将Deep空间的解转换为原始空间的完整解
"""

import numpy as np
import math
import random
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
import copy


class DeepSpaceDecoder:
    """Deep空间解码器"""
    
    def __init__(self, enhanced_constructor, instance):
        """
        初始化解码器
        
        Args:
            enhanced_constructor: 增强的Deep空间构造器
            instance: 原始VRPTW实例
        """
        self.constructor = enhanced_constructor
        self.instance = instance
        self.deep_nodes = enhanced_constructor.deep_nodes
        self.node_mapping = enhanced_constructor.node_mapping
        self.deep_instance_data = enhanced_constructor.get_deep_instance_data()
        
        # 解码策略配置
        self.decoding_strategies = {
            'nearest_neighbor': self._decode_nearest_neighbor,
            'time_oriented': self._decode_time_oriented,
            'savings': self._decode_savings,
            'hybrid': self._decode_hybrid
        }
    
    def decode_deep_solution(self, deep_solution: List[List[int]], 
                           strategy: str = 'hybrid') -> List[List[int]]:
        """
        将Deep空间解码为原始空间解
        
        Args:
            deep_solution: Deep空间的解（路径列表）
            strategy: 解码策略
            
        Returns:
            原始空间的解（路径列表）
        """
        if strategy not in self.decoding_strategies:
            raise ValueError(f"未知的解码策略: {strategy}")
        
        return self.decoding_strategies[strategy](deep_solution)
    
    def _decode_nearest_neighbor(self, deep_solution: List[List[int]]) -> List[List[int]]:
        """最近邻解码策略"""
        original_solution = []
        
        for deep_route in deep_solution:
            if not deep_route:
                continue
            
            # 为每个Deep路径解码
            original_route = self._decode_single_route_nn(deep_route)
            if original_route:
                original_solution.append(original_route)
        
        return original_solution
    
    def _decode_single_route_nn(self, deep_route: List[int]) -> List[int]:
        """使用最近邻策略解码单条路径"""
        if not deep_route:
            return []
        
        original_route = []
        current_time = 0
        current_load = 0
        current_x, current_y = self.instance.depot['x'], self.instance.depot['y']
        
        for deep_node_id in deep_route:
            if deep_node_id >= len(self.deep_nodes):
                continue
            
            deep_node = self.deep_nodes[deep_node_id]
            customers = deep_node.customers.copy()
            
            # 检查容量约束
            if current_load + deep_node.total_demand > self.instance.vehicle_capacity:
                break
            
            # 解码聚类内的客户顺序
            cluster_route = self._solve_cluster_tsp(customers, current_x, current_y, current_time)
            
            # 验证时间窗约束
            valid_route, new_time, new_x, new_y = self._validate_cluster_route(
                cluster_route, current_x, current_y, current_time)
            
            if valid_route:
                original_route.extend(valid_route)
                current_time = new_time
                current_load += deep_node.total_demand
                current_x, current_y = new_x, new_y
            else:
                # 如果无法满足约束，尝试部分插入
                partial_route = self._partial_insert_cluster(
                    customers, current_x, current_y, current_time, 
                    self.instance.vehicle_capacity - current_load)
                if partial_route:
                    original_route.extend(partial_route)
                break
        
        return original_route
    
    def _decode_time_oriented(self, deep_solution: List[List[int]]) -> List[List[int]]:
        """时间导向解码策略"""
        original_solution = []
        
        for deep_route in deep_solution:
            if not deep_route:
                continue
            
            # 按时间窗排序Deep节点
            sorted_deep_route = sorted(deep_route, 
                                     key=lambda x: self.deep_nodes[x].ready_time if x < len(self.deep_nodes) else float('inf'))
            
            original_route = self._decode_single_route_time_oriented(sorted_deep_route)
            if original_route:
                original_solution.append(original_route)
        
        return original_solution
    
    def _decode_single_route_time_oriented(self, deep_route: List[int]) -> List[int]:
        """使用时间导向策略解码单条路径"""
        if not deep_route:
            return []
        
        original_route = []
        current_time = 0
        current_load = 0
        current_x, current_y = self.instance.depot['x'], self.instance.depot['y']
        
        for deep_node_id in deep_route:
            if deep_node_id >= len(self.deep_nodes):
                continue
            
            deep_node = self.deep_nodes[deep_node_id]
            customers = deep_node.customers.copy()
            
            # 检查容量约束
            if current_load + deep_node.total_demand > self.instance.vehicle_capacity:
                break
            
            # 按时间窗排序客户
            customers.sort(key=lambda c: self.instance.customers[c]['ready_time'])
            
            # 验证并调整客户顺序
            cluster_route = self._optimize_cluster_time_windows(
                customers, current_x, current_y, current_time)
            
            if cluster_route:
                # 验证路径
                valid_route, new_time, new_x, new_y = self._validate_cluster_route(
                    cluster_route, current_x, current_y, current_time)
                
                if valid_route:
                    original_route.extend(valid_route)
                    current_time = new_time
                    current_load += sum(self.instance.customers[c]['demand'] for c in valid_route)
                    current_x, current_y = new_x, new_y
                else:
                    break
        
        return original_route
    
    def _decode_savings(self, deep_solution: List[List[int]]) -> List[List[int]]:
        """节约算法解码策略"""
        original_solution = []
        
        for deep_route in deep_solution:
            if not deep_route:
                continue
            
            original_route = self._decode_single_route_savings(deep_route)
            if original_route:
                original_solution.append(original_route)
        
        return original_solution
    
    def _decode_single_route_savings(self, deep_route: List[int]) -> List[int]:
        """使用节约算法解码单条路径"""
        if not deep_route:
            return []
        
        # 收集所有客户
        all_customers = []
        for deep_node_id in deep_route:
            if deep_node_id < len(self.deep_nodes):
                all_customers.extend(self.deep_nodes[deep_node_id].customers)
        
        if not all_customers:
            return []
        
        # 使用节约算法构造路径
        return self._savings_algorithm(all_customers)
    
    def _decode_hybrid(self, deep_solution: List[List[int]]) -> List[List[int]]:
        """混合解码策略"""
        original_solution = []
        
        for deep_route in deep_solution:
            if not deep_route:
                continue
            
            # 根据路径特征选择解码策略
            strategy = self._select_decoding_strategy(deep_route)
            
            if strategy == 'time_oriented':
                original_route = self._decode_single_route_time_oriented(deep_route)
            elif strategy == 'savings':
                original_route = self._decode_single_route_savings(deep_route)
            else:
                original_route = self._decode_single_route_nn(deep_route)
            
            if original_route:
                original_solution.append(original_route)
        
        return original_solution
    
    def _select_decoding_strategy(self, deep_route: List[int]) -> str:
        """根据路径特征选择解码策略"""
        if not deep_route:
            return 'nearest_neighbor'
        
        # 计算路径特征
        total_customers = 0
        avg_urgency = 0
        total_demand = 0
        
        for deep_node_id in deep_route:
            if deep_node_id < len(self.deep_nodes):
                deep_node = self.deep_nodes[deep_node_id]
                total_customers += deep_node.customer_count
                avg_urgency += deep_node.urgency
                total_demand += deep_node.total_demand
        
        if len(deep_route) > 0:
            avg_urgency /= len(deep_route)
        
        # 策略选择逻辑
        if avg_urgency > 0.7:  # 高紧急度
            return 'time_oriented'
        elif total_customers > 15:  # 客户数量多
            return 'savings'
        else:
            return 'nearest_neighbor'
    
    def _solve_cluster_tsp(self, customers: List[int], start_x: float, start_y: float, 
                          start_time: float) -> List[int]:
        """解决聚类内的TSP问题"""
        if len(customers) <= 1:
            return customers
        
        # 使用最近邻启发式
        unvisited = set(customers)
        route = []
        current_x, current_y = start_x, start_y
        
        while unvisited:
            # 找到最近的客户
            nearest = min(unvisited, key=lambda c: self._distance_to_customer(current_x, current_y, c))
            route.append(nearest)
            unvisited.remove(nearest)
            current_x = self.instance.customers[nearest]['x']
            current_y = self.instance.customers[nearest]['y']
        
        return route
    
    def _optimize_cluster_time_windows(self, customers: List[int], start_x: float, 
                                     start_y: float, start_time: float) -> List[int]:
        """优化聚类内客户的时间窗顺序"""
        if len(customers) <= 1:
            return customers
        
        # 计算每个客户的最早可达时间
        customer_earliest_times = {}
        for customer in customers:
            travel_time = self._distance_to_customer(start_x, start_y, customer)
            earliest_arrival = start_time + travel_time
            ready_time = self.instance.customers[customer]['ready_time']
            customer_earliest_times[customer] = max(earliest_arrival, ready_time)
        
        # 按最早可达时间排序
        sorted_customers = sorted(customers, key=lambda c: customer_earliest_times[c])
        
        # 验证时间窗可行性
        current_time = start_time
        current_x, current_y = start_x, start_y
        feasible_route = []
        
        for customer in sorted_customers:
            travel_time = self._distance_to_customer(current_x, current_y, customer)
            arrival_time = current_time + travel_time
            ready_time = self.instance.customers[customer]['ready_time']
            due_time = self.instance.customers[customer]['due_time']
            service_time = self.instance.customers[customer]['service_time']
            
            # 检查时间窗约束
            if arrival_time <= due_time:
                service_start = max(arrival_time, ready_time)
                feasible_route.append(customer)
                current_time = service_start + service_time
                current_x = self.instance.customers[customer]['x']
                current_y = self.instance.customers[customer]['y']
            else:
                # 时间窗不可行，尝试插入到更早的位置
                break
        
        return feasible_route
    
    def _validate_cluster_route(self, route: List[int], start_x: float, start_y: float, 
                              start_time: float) -> Tuple[List[int], float, float, float]:
        """验证聚类路径的可行性"""
        if not route:
            return [], start_time, start_x, start_y
        
        current_time = start_time
        current_x, current_y = start_x, start_y
        valid_route = []
        
        for customer in route:
            travel_time = self._distance_to_customer(current_x, current_y, customer)
            arrival_time = current_time + travel_time
            ready_time = self.instance.customers[customer]['ready_time']
            due_time = self.instance.customers[customer]['due_time']
            service_time = self.instance.customers[customer]['service_time']
            
            # 检查时间窗约束
            if arrival_time <= due_time:
                service_start = max(arrival_time, ready_time)
                valid_route.append(customer)
                current_time = service_start + service_time
                current_x = self.instance.customers[customer]['x']
                current_y = self.instance.customers[customer]['y']
            else:
                break
        
        return valid_route, current_time, current_x, current_y
    
    def _partial_insert_cluster(self, customers: List[int], start_x: float, start_y: float, 
                              start_time: float, remaining_capacity: float) -> List[int]:
        """部分插入聚类客户"""
        # 按需求排序，优先插入小需求客户
        sorted_customers = sorted(customers, key=lambda c: self.instance.customers[c]['demand'])
        
        partial_route = []
        current_capacity = 0
        current_time = start_time
        current_x, current_y = start_x, start_y
        
        for customer in sorted_customers:
            demand = self.instance.customers[customer]['demand']
            
            # 检查容量约束
            if current_capacity + demand > remaining_capacity:
                continue
            
            # 检查时间窗约束
            travel_time = self._distance_to_customer(current_x, current_y, customer)
            arrival_time = current_time + travel_time
            ready_time = self.instance.customers[customer]['ready_time']
            due_time = self.instance.customers[customer]['due_time']
            service_time = self.instance.customers[customer]['service_time']
            
            if arrival_time <= due_time:
                service_start = max(arrival_time, ready_time)
                partial_route.append(customer)
                current_capacity += demand
                current_time = service_start + service_time
                current_x = self.instance.customers[customer]['x']
                current_y = self.instance.customers[customer]['y']
        
        return partial_route
    
    def _savings_algorithm(self, customers: List[int]) -> List[int]:
        """节约算法构造路径"""
        if len(customers) <= 1:
            return customers
        
        # 计算节约值
        savings = []
        depot_x, depot_y = self.instance.depot['x'], self.instance.depot['y']
        
        for i in range(len(customers)):
            for j in range(i + 1, len(customers)):
                c1, c2 = customers[i], customers[j]
                
                # 计算节约值: s(i,j) = d(0,i) + d(0,j) - d(i,j)
                dist_depot_c1 = self._distance_to_customer(depot_x, depot_y, c1)
                dist_depot_c2 = self._distance_to_customer(depot_x, depot_y, c2)
                dist_c1_c2 = self._distance_between_customers(c1, c2)
                
                saving = dist_depot_c1 + dist_depot_c2 - dist_c1_c2
                savings.append((saving, c1, c2))
        
        # 按节约值降序排序
        savings.sort(reverse=True)
        
        # 构造路径
        routes = {c: [c] for c in customers}  # 每个客户初始为单独路径
        
        for saving, c1, c2 in savings:
            # 检查是否可以合并路径
            route1 = routes[c1]
            route2 = routes[c2]
            
            if route1 != route2 and self._can_merge_routes(route1, route2, c1, c2):
                # 合并路径
                merged_route = self._merge_routes(route1, route2, c1, c2)
                
                # 更新路径映射
                for c in merged_route:
                    routes[c] = merged_route
        
        # 返回最长的路径（包含所有客户）
        return max(routes.values(), key=len)
    
    def _can_merge_routes(self, route1: List[int], route2: List[int], 
                         c1: int, c2: int) -> bool:
        """检查是否可以合并两条路径"""
        # 检查容量约束
        total_demand = sum(self.instance.customers[c]['demand'] for c in route1 + route2)
        if total_demand > self.instance.vehicle_capacity:
            return False
        
        # 检查连接点是否在路径端点
        if (c1 not in [route1[0], route1[-1]]) or (c2 not in [route2[0], route2[-1]]):
            return False
        
        return True
    
    def _merge_routes(self, route1: List[int], route2: List[int], 
                     c1: int, c2: int) -> List[int]:
        """合并两条路径"""
        # 确定合并方向
        if c1 == route1[0] and c2 == route2[0]:
            return list(reversed(route1)) + route2
        elif c1 == route1[0] and c2 == route2[-1]:
            return route2 + route1
        elif c1 == route1[-1] and c2 == route2[0]:
            return route1 + route2
        elif c1 == route1[-1] and c2 == route2[-1]:
            return route1 + list(reversed(route2))
        else:
            return route1 + route2  # 默认合并
    
    def _distance_to_customer(self, x: float, y: float, customer: int) -> float:
        """计算到客户的距离"""
        cx = self.instance.customers[customer]['x']
        cy = self.instance.customers[customer]['y']
        return math.sqrt((x - cx)**2 + (y - cy)**2)
    
    def _distance_between_customers(self, c1: int, c2: int) -> float:
        """计算两个客户之间的距离"""
        x1 = self.instance.customers[c1]['x']
        y1 = self.instance.customers[c1]['y']
        x2 = self.instance.customers[c2]['x']
        y2 = self.instance.customers[c2]['y']
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def evaluate_solution(self, solution: List[List[int]]) -> Dict[str, Any]:
        """评估解的质量"""
        total_distance = 0
        total_vehicles = len(solution)
        total_customers = 0
        feasible = True
        violations = []
        
        depot_x, depot_y = self.instance.depot['x'], self.instance.depot['y']
        
        for route_idx, route in enumerate(solution):
            if not route:
                continue
            
            route_distance = 0
            route_load = 0
            current_time = 0
            current_x, current_y = depot_x, depot_y
            
            # 检查路径
            for customer in route:
                # 计算旅行时间和距离
                travel_dist = self._distance_to_customer(current_x, current_y, customer)
                route_distance += travel_dist
                
                arrival_time = current_time + travel_dist
                ready_time = self.instance.customers[customer]['ready_time']
                due_time = self.instance.customers[customer]['due_time']
                service_time = self.instance.customers[customer]['service_time']
                demand = self.instance.customers[customer]['demand']
                
                # 检查时间窗约束
                if arrival_time > due_time:
                    feasible = False
                    violations.append(f"路径{route_idx}客户{customer}时间窗违反")
                
                # 检查容量约束
                route_load += demand
                if route_load > self.instance.vehicle_capacity:
                    feasible = False
                    violations.append(f"路径{route_idx}容量约束违反")
                
                # 更新状态
                service_start = max(arrival_time, ready_time)
                current_time = service_start + service_time
                current_x = self.instance.customers[customer]['x']
                current_y = self.instance.customers[customer]['y']
                total_customers += 1
            
            # 返回仓库
            route_distance += self._distance_to_customer(current_x, current_y, 0)  # 假设仓库为客户0
            total_distance += route_distance
        
        return {
            'total_distance': total_distance,
            'total_vehicles': total_vehicles,
            'total_customers': total_customers,
            'feasible': feasible,
            'violations': violations,
            'objective': total_distance  # 可以根据需要调整目标函数
        }
    
    def improve_solution(self, solution: List[List[int]], max_iterations: int = 100) -> List[List[int]]:
        """改进解的质量"""
        current_solution = copy.deepcopy(solution)
        current_eval = self.evaluate_solution(current_solution)
        
        for iteration in range(max_iterations):
            # 尝试2-opt改进
            improved_solution = self._apply_2opt(current_solution)
            improved_eval = self.evaluate_solution(improved_solution)
            
            if improved_eval['objective'] < current_eval['objective']:
                current_solution = improved_solution
                current_eval = improved_eval
            
            # 尝试relocate改进
            improved_solution = self._apply_relocate(current_solution)
            improved_eval = self.evaluate_solution(improved_solution)
            
            if improved_eval['objective'] < current_eval['objective']:
                current_solution = improved_solution
                current_eval = improved_eval
        
        return current_solution
    
    def _apply_2opt(self, solution: List[List[int]]) -> List[List[int]]:
        """应用2-opt改进"""
        improved_solution = copy.deepcopy(solution)
        
        for route_idx, route in enumerate(improved_solution):
            if len(route) < 4:
                continue
            
            best_route = route.copy()
            best_distance = self._calculate_route_distance(route)
            
            for i in range(len(route) - 1):
                for j in range(i + 2, len(route)):
                    # 执行2-opt交换
                    new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                    new_distance = self._calculate_route_distance(new_route)
                    
                    if new_distance < best_distance:
                        best_route = new_route
                        best_distance = new_distance
            
            improved_solution[route_idx] = best_route
        
        return improved_solution
    
    def _apply_relocate(self, solution: List[List[int]]) -> List[List[int]]:
        """应用relocate改进"""
        improved_solution = copy.deepcopy(solution)
        
        for route_idx, route in enumerate(improved_solution):
            if len(route) <= 1:
                continue
            
            best_route = route.copy()
            best_distance = self._calculate_route_distance(route)
            
            for i in range(len(route)):
                for j in range(len(route)):
                    if i == j:
                        continue
                    
                    # 移动客户i到位置j
                    new_route = route.copy()
                    customer = new_route.pop(i)
                    new_route.insert(j if j < i else j-1, customer)
                    
                    new_distance = self._calculate_route_distance(new_route)
                    
                    if new_distance < best_distance:
                        best_route = new_route
                        best_distance = new_distance
            
            improved_solution[route_idx] = best_route
        
        return improved_solution
    
    def _calculate_route_distance(self, route: List[int]) -> float:
        """计算路径距离"""
        if not route:
            return 0
        
        distance = 0
        depot_x, depot_y = self.instance.depot['x'], self.instance.depot['y']
        
        # 从仓库到第一个客户
        distance += self._distance_to_customer(depot_x, depot_y, route[0])
        
        # 客户之间的距离
        for i in range(len(route) - 1):
            distance += self._distance_between_customers(route[i], route[i+1])
        
        # 从最后一个客户回到仓库
        distance += self._distance_to_customer(
            self.instance.customers[route[-1]]['x'],
            self.instance.customers[route[-1]]['y'], 0)
        
        return distance


# 使用示例
if __name__ == "__main__":
    from vrptw_instance import VRPTWInstance
    from enhanced_deep_constructor import EnhancedDeepConstructor
    
    # 加载实例
    instance = VRPTWInstance("data/RC1_2_1.txt")
    
    # 创建增强的Deep空间构造器
    constructor = EnhancedDeepConstructor(instance, cluster_method='time_spatial')
    deep_nodes = constructor.construct_deep_space(target_clusters=8)
    
    # 创建解码器
    decoder = DeepSpaceDecoder(constructor, instance)
    
    # 示例Deep空间解
    deep_solution = [[0, 1, 2], [3, 4], [5, 6, 7]]
    
    # 解码为原始空间解
    original_solution = decoder.decode_deep_solution(deep_solution, strategy='hybrid')
    
    print("Deep空间解:", deep_solution)
    print("原始空间解:", original_solution)
    
    # 评估解质量
    evaluation = decoder.evaluate_solution(original_solution)
    print("解评估:", evaluation)
    
    # 改进解
    improved_solution = decoder.improve_solution(original_solution)
    improved_evaluation = decoder.evaluate_solution(improved_solution)
    print("改进后解评估:", improved_evaluation)