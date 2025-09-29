# local_search_operators.py
# VRPTW局部搜索算子模块

import numpy as np
import copy
from typing import List, Dict, Tuple, Any, Optional, Callable
from abc import ABC, abstractmethod


class LocalSearchOperator(ABC):
    """
    局部搜索算子基类，定义了所有算子必须实现的接口
    """
    
    def __init__(self, instance, distance_matrix=None, time_windows=None, demands=None, capacity=None):
        """
        初始化局部搜索算子
        
        Args:
            instance: VRPTW实例对象
            distance_matrix: 距离矩阵，如果为None则从instance中获取
            time_windows: 时间窗约束，如果为None则从instance中获取
            demands: 客户需求，如果为None则从instance中获取
            capacity: 车辆容量，如果为None则从instance中获取
        """
        self.instance = instance
        self.distance_matrix = distance_matrix if distance_matrix is not None else instance.distance_matrix
        
        # 提取时间窗信息
        if time_windows is None:
            self.time_windows = []
            for i in range(len(instance.ordinary_customers)):
                customer = instance.ordinary_customers[i]
                self.time_windows.append((customer['ready_time'], customer['due_date']))
        else:
            self.time_windows = time_windows
            
        # 提取需求信息
        if demands is None:
            self.demands = [customer['demand'] for customer in instance.ordinary_customers]
        else:
            self.demands = demands
            
        # 提取容量信息
        self.capacity = capacity if capacity is not None else instance.vehicle_capacity
        
        # 服务时间
        self.service_times = [customer['service_time'] for customer in instance.ordinary_customers]
        
    @abstractmethod
    def apply(self, routes: List[List[int]], **kwargs) -> Tuple[List[List[int]], bool, float]:
        """
        应用局部搜索算子到给定的路径集合
        
        Args:
            routes: 路径列表，每个路径是客户索引列表（0-based）
            **kwargs: 其他参数
            
        Returns:
            Tuple[List[List[int]], bool, float]: 
                - 优化后的路径列表
                - 是否有改进
                - 改进的成本差值
        """
        pass
    
    def check_time_window_feasibility(self, route: List[int]) -> bool:
        """
        检查路径的时间窗可行性
        
        Args:
            route: 客户索引列表（0-based）
            
        Returns:
            bool: 路径是否满足时间窗约束
        """
        current_time = 0  # 从仓库出发的时间
        prev_node = 0  # 仓库索引
        
        for customer in route:
            # 计算到达时间 = 当前时间 + 行驶时间
            travel_time = self.distance_matrix[prev_node][customer + 1]  # +1因为customer是0-based
            arrival_time = current_time + travel_time
            
            # 获取客户时间窗
            ready_time, due_date = self.time_windows[customer]
            
            # 如果早到，需要等待
            start_service_time = max(arrival_time, ready_time)
            
            # 检查是否超过截止时间
            if start_service_time > due_date:
                return False
            
            # 更新当前时间和前一个节点
            service_time = self.service_times[customer]
            current_time = start_service_time + service_time
            prev_node = customer + 1
        
        # 返回仓库
        travel_time = self.distance_matrix[prev_node][0]
        final_time = current_time + travel_time
        
        # 检查是否在仓库时间窗内返回（如果有限制）
        # 这里假设仓库没有时间窗限制，如果有，可以在这里添加检查
        
        return True
    
    def check_capacity_feasibility(self, route: List[int]) -> bool:
        """
        检查路径的容量可行性
        
        Args:
            route: 客户索引列表（0-based）
            
        Returns:
            bool: 路径是否满足容量约束
        """
        total_demand = sum(self.demands[customer] for customer in route)
        return total_demand <= self.capacity
    
    def calculate_route_cost(self, route: List[int]) -> float:
        """
        计算路径的总距离成本
        
        Args:
            route: 客户索引列表（0-based）
            
        Returns:
            float: 路径的总距离
        """
        if not route:
            return 0.0
        
        cost = 0.0
        prev_node = 0  # 从仓库出发
        
        for customer in route:
            cost += self.distance_matrix[prev_node][customer + 1]
            prev_node = customer + 1
            
        # 返回仓库
        cost += self.distance_matrix[prev_node][0]
        
        return cost
    
    def calculate_total_cost(self, routes: List[List[int]]) -> float:
        """
        计算所有路径的总成本
        
        Args:
            routes: 路径列表，每个路径是客户索引列表
            
        Returns:
            float: 总成本
        """
        return sum(self.calculate_route_cost(route) for route in routes)
    
    def is_feasible_solution(self, routes: List[List[int]]) -> bool:
        """
        检查解决方案是否可行（满足所有约束）
        
        Args:
            routes: 路径列表，每个路径是客户索引列表
            
        Returns:
            bool: 解决方案是否可行
        """
        for route in routes:
            if not self.check_capacity_feasibility(route):
                return False
            if not self.check_time_window_feasibility(route):
                return False
        return True


class RelocateOperator(LocalSearchOperator):
    """
    Relocate（插入）算子：从某路线摘除一个客户节点，插入到同一路线或另一条路线的其它位置
    """
    
    def __init__(self, instance, distance_matrix=None, time_windows=None, demands=None, capacity=None):
        """初始化Relocate算子"""
        super().__init__(instance, distance_matrix, time_windows, demands, capacity)
    
    def apply(self, routes: List[List[int]], **kwargs) -> Tuple[List[List[int]], bool, float]:
        """
        应用Relocate算子到给定的路径集合
        
        Args:
            routes: 路径列表，每个路径是客户索引列表（0-based）
            **kwargs: 其他参数，可包含：
                - max_trials: 最大尝试次数
                - first_improvement: 是否采用第一次改进策略
                
        Returns:
            Tuple[List[List[int]], bool, float]: 
                - 优化后的路径列表
                - 是否有改进
                - 改进的成本差值
        """
        max_trials = kwargs.get('max_trials', float('inf'))
        first_improvement = kwargs.get('first_improvement', True)
        
        # 复制路径以避免修改原始数据
        best_routes = copy.deepcopy(routes)
        original_cost = self.calculate_total_cost(routes)
        best_cost = original_cost
        improved = False
        trials = 0
        
        # 遍历所有路径和客户
        for source_route_idx, source_route in enumerate(routes):
            if not source_route:  # 跳过空路径
                continue
                
            for source_pos, customer in enumerate(source_route):
                # 尝试将客户从source_route移除，并插入到其他位置
                
                # 创建移除客户后的路径
                new_source_route = source_route[:source_pos] + source_route[source_pos+1:]
                
                # 尝试插入到所有可能的位置
                for target_route_idx, target_route in enumerate(routes):
                    # 确定插入位置范围
                    max_pos = len(target_route) + 1 if target_route_idx != source_route_idx else len(target_route)
                    
                    for target_pos in range(max_pos):
                        # 跳过原位置
                        if target_route_idx == source_route_idx and (target_pos == source_pos or target_pos == source_pos + 1):
                            continue
                        
                        trials += 1
                        if trials > max_trials:
                            return best_routes, improved, original_cost - best_cost
                        
                        # 创建新的目标路径
                        new_target_route = target_route.copy()
                        new_target_route.insert(target_pos, customer)
                        
                        # 创建新的路径集合
                        new_routes = copy.deepcopy(routes)
                        new_routes[source_route_idx] = new_source_route
                        new_routes[target_route_idx] = new_target_route
                        
                        # 检查可行性
                        if not self.check_capacity_feasibility(new_target_route):
                            continue
                        if not self.check_time_window_feasibility(new_target_route):
                            continue
                        if source_route_idx != target_route_idx and not self.check_time_window_feasibility(new_source_route):
                            continue
                        
                        # 计算新成本
                        new_cost = self.calculate_total_cost(new_routes)
                        
                        # 如果有改进，更新最佳解
                        if new_cost < best_cost:
                            best_routes = new_routes
                            best_cost = new_cost
                            improved = True
                            
                            # 如果采用第一次改进策略，立即返回
                            if first_improvement:
                                return best_routes, True, original_cost - best_cost
        
        return best_routes, improved, original_cost - best_cost


class SwapOperator(LocalSearchOperator):
    """
    Swap（交换）算子：交换两个客户节点的位置（可以在同一路径，也可以跨路径）
    """
    
    def __init__(self, instance, distance_matrix=None, time_windows=None, demands=None, capacity=None):
        """初始化Swap算子"""
        super().__init__(instance, distance_matrix, time_windows, demands, capacity)
    
    def apply(self, routes: List[List[int]], **kwargs) -> Tuple[List[List[int]], bool, float]:
        """
        应用Swap算子到给定的路径集合
        
        Args:
            routes: 路径列表，每个路径是客户索引列表（0-based）
            **kwargs: 其他参数，可包含：
                - max_trials: 最大尝试次数
                - first_improvement: 是否采用第一次改进策略
                
        Returns:
            Tuple[List[List[int]], bool, float]: 
                - 优化后的路径列表
                - 是否有改进
                - 改进的成本差值
        """
        max_trials = kwargs.get('max_trials', float('inf'))
        first_improvement = kwargs.get('first_improvement', True)
        
        # 复制路径以避免修改原始数据
        best_routes = copy.deepcopy(routes)
        original_cost = self.calculate_total_cost(routes)
        best_cost = original_cost
        improved = False
        trials = 0
        
        # 遍历所有可能的交换对
        for route1_idx, route1 in enumerate(routes):
            if not route1:  # 跳过空路径
                continue
                
            for pos1, customer1 in enumerate(route1):
                for route2_idx, route2 in enumerate(routes[route1_idx:], route1_idx):
                    # 确定起始位置（避免重复）
                    start_pos = pos1 + 1 if route1_idx == route2_idx else 0
                    
                    for pos2, customer2 in enumerate(route2[start_pos:], start_pos):
                        trials += 1
                        if trials > max_trials:
                            return best_routes, improved, original_cost - best_cost
                        
                        # 创建新的路径
                        new_route1 = route1.copy()
                        new_route2 = route2.copy()
                        
                        # 交换客户
                        new_route1[pos1] = customer2
                        new_route2[pos2] = customer1
                        
                        # 创建新的路径集合
                        new_routes = copy.deepcopy(routes)
                        new_routes[route1_idx] = new_route1
                        new_routes[route2_idx] = new_route2
                        
                        # 检查可行性
                        if not self.check_capacity_feasibility(new_route1) or not self.check_capacity_feasibility(new_route2):
                            continue
                        if not self.check_time_window_feasibility(new_route1) or not self.check_time_window_feasibility(new_route2):
                            continue
                        
                        # 计算新成本
                        new_cost = self.calculate_total_cost(new_routes)
                        
                        # 如果有改进，更新最佳解
                        if new_cost < best_cost:
                            best_routes = new_routes
                            best_cost = new_cost
                            improved = True
                            
                            # 如果采用第一次改进策略，立即返回
                            if first_improvement:
                                return best_routes, True, original_cost - best_cost
        
        return best_routes, improved, original_cost - best_cost


class OrOptOperator(LocalSearchOperator):
    """
    Or-opt（子串移动）算子：在同一路线或不同路线中，把一段连续的子串移动到另一个位置
    """
    
    def __init__(self, instance, distance_matrix=None, time_windows=None, demands=None, capacity=None, segment_length=2):
        """
        初始化Or-opt算子
        
        Args:
            segment_length: 子串长度，默认为2
        """
        super().__init__(instance, distance_matrix, time_windows, demands, capacity)
        self.segment_length = segment_length
    
    def apply(self, routes: List[List[int]], **kwargs) -> Tuple[List[List[int]], bool, float]:
        """
        应用Or-opt算子到给定的路径集合
        
        Args:
            routes: 路径列表，每个路径是客户索引列表（0-based）
            **kwargs: 其他参数，可包含：
                - max_trials: 最大尝试次数
                - first_improvement: 是否采用第一次改进策略
                - segment_length: 子串长度，覆盖初始化时设置的值
                
        Returns:
            Tuple[List[List[int]], bool, float]: 
                - 优化后的路径列表
                - 是否有改进
                - 改进的成本差值
        """
        max_trials = kwargs.get('max_trials', float('inf'))
        first_improvement = kwargs.get('first_improvement', True)
        segment_length = kwargs.get('segment_length', self.segment_length)
        
        # 复制路径以避免修改原始数据
        best_routes = copy.deepcopy(routes)
        original_cost = self.calculate_total_cost(routes)
        best_cost = original_cost
        improved = False
        trials = 0
        
        # 遍历所有路径
        for source_route_idx, source_route in enumerate(routes):
            if len(source_route) < segment_length:  # 跳过长度不足的路径
                continue
                
            # 遍历所有可能的子串
            for start_pos in range(len(source_route) - segment_length + 1):
                # 提取子串
                segment = source_route[start_pos:start_pos + segment_length]
                
                # 创建移除子串后的路径
                new_source_route = source_route[:start_pos] + source_route[start_pos + segment_length:]
                
                # 尝试插入到所有可能的位置
                for target_route_idx, target_route in enumerate(routes):
                    # 确定插入位置范围
                    max_pos = len(target_route) + 1
                    
                    for target_pos in range(max_pos):
                        # 跳过原位置附近
                        if target_route_idx == source_route_idx:
                            if target_pos > start_pos and target_pos <= start_pos + segment_length:
                                continue
                        
                        trials += 1
                        if trials > max_trials:
                            return best_routes, improved, original_cost - best_cost
                        
                        # 创建新的目标路径
                        new_target_route = target_route.copy()
                        for i, customer in enumerate(segment):
                            new_target_route.insert(target_pos + i, customer)
                        
                        # 创建新的路径集合
                        new_routes = copy.deepcopy(routes)
                        new_routes[source_route_idx] = new_source_route
                        new_routes[target_route_idx] = new_target_route
                        
                        # 检查可行性
                        if not self.check_capacity_feasibility(new_target_route):
                            continue
                        if not self.check_time_window_feasibility(new_target_route):
                            continue
                        if source_route_idx != target_route_idx and not self.check_time_window_feasibility(new_source_route):
                            continue
                        
                        # 计算新成本
                        new_cost = self.calculate_total_cost(new_routes)
                        
                        # 如果有改进，更新最佳解
                        if new_cost < best_cost:
                            best_routes = new_routes
                            best_cost = new_cost
                            improved = True
                            
                            # 如果采用第一次改进策略，立即返回
                            if first_improvement:
                                return best_routes, True, original_cost - best_cost
        
        return best_routes, improved, original_cost - best_cost


class CrossExchangeOperator(LocalSearchOperator):
    """
    Cross-exchange（子串交换）算子：在两条路线之间交换两个子串
    """
    
    def __init__(self, instance, distance_matrix=None, time_windows=None, demands=None, capacity=None, 
                 max_segment_length=3):
        """
        初始化Cross-exchange算子
        
        Args:
            max_segment_length: 最大子串长度，默认为3
        """
        super().__init__(instance, distance_matrix, time_windows, demands, capacity)
        self.max_segment_length = max_segment_length
    
    def apply(self, routes: List[List[int]], **kwargs) -> Tuple[List[List[int]], bool, float]:
        """
        应用Cross-exchange算子到给定的路径集合
        
        Args:
            routes: 路径列表，每个路径是客户索引列表（0-based）
            **kwargs: 其他参数，可包含：
                - max_trials: 最大尝试次数
                - first_improvement: 是否采用第一次改进策略
                - max_segment_length: 最大子串长度，覆盖初始化时设置的值
                
        Returns:
            Tuple[List[List[int]], bool, float]: 
                - 优化后的路径列表
                - 是否有改进
                - 改进的成本差值
        """
        max_trials = kwargs.get('max_trials', float('inf'))
        first_improvement = kwargs.get('first_improvement', True)
        max_segment_length = kwargs.get('max_segment_length', self.max_segment_length)
        
        # 复制路径以避免修改原始数据
        best_routes = copy.deepcopy(routes)
        original_cost = self.calculate_total_cost(routes)
        best_cost = original_cost
        improved = False
        trials = 0
        
        # 遍历所有路径对
        for route1_idx in range(len(routes)):
            route1 = routes[route1_idx]
            if len(route1) < 1:  # 跳过空路径
                continue
                
            for route2_idx in range(route1_idx + 1, len(routes)):
                route2 = routes[route2_idx]
                if len(route2) < 1:  # 跳过空路径
                    continue
                
                # 尝试不同长度的子串
                for len1 in range(1, min(max_segment_length, len(route1)) + 1):
                    for len2 in range(1, min(max_segment_length, len(route2)) + 1):
                        
                        # 遍历所有可能的子串位置
                        for pos1 in range(len(route1) - len1 + 1):
                            for pos2 in range(len(route2) - len2 + 1):
                                trials += 1
                                if trials > max_trials:
                                    return best_routes, improved, original_cost - best_cost
                                
                                # 提取子串
                                segment1 = route1[pos1:pos1 + len1]
                                segment2 = route2[pos2:pos2 + len2]
                                
                                # 创建新的路径
                                new_route1 = route1[:pos1] + segment2 + route1[pos1 + len1:]
                                new_route2 = route2[:pos2] + segment1 + route2[pos2 + len2:]
                                
                                # 创建新的路径集合
                                new_routes = copy.deepcopy(routes)
                                new_routes[route1_idx] = new_route1
                                new_routes[route2_idx] = new_route2
                                
                                # 检查可行性
                                if not self.check_capacity_feasibility(new_route1) or not self.check_capacity_feasibility(new_route2):
                                    continue
                                if not self.check_time_window_feasibility(new_route1) or not self.check_time_window_feasibility(new_route2):
                                    continue
                                
                                # 计算新成本
                                new_cost = self.calculate_total_cost(new_routes)
                                
                                # 如果有改进，更新最佳解
                                if new_cost < best_cost:
                                    best_routes = new_routes
                                    best_cost = new_cost
                                    improved = True
                                    
                                    # 如果采用第一次改进策略，立即返回
                                    if first_improvement:
                                        return best_routes, True, original_cost - best_cost
        
        return best_routes, improved, original_cost - best_cost


class TwoOptOperator(LocalSearchOperator):
    """
    2-opt（边交换）算子：对一条路线内部做断开 + 重接
    """
    
    def __init__(self, instance, distance_matrix=None, time_windows=None, demands=None, capacity=None):
        """初始化2-opt算子"""
        super().__init__(instance, distance_matrix, time_windows, demands, capacity)
    
    def apply(self, routes: List[List[int]], **kwargs) -> Tuple[List[List[int]], bool, float]:
        """
        应用2-opt算子到给定的路径集合
        
        Args:
            routes: 路径列表，每个路径是客户索引列表（0-based）
            **kwargs: 其他参数，可包含：
                - max_trials: 最大尝试次数
                - first_improvement: 是否采用第一次改进策略
                - intra_route_only: 是否只在路径内部应用，默认为True
                
        Returns:
            Tuple[List[List[int]], bool, float]: 
                - 优化后的路径列表
                - 是否有改进
                - 改进的成本差值
        """
        max_trials = kwargs.get('max_trials', float('inf'))
        first_improvement = kwargs.get('first_improvement', True)
        intra_route_only = kwargs.get('intra_route_only', True)
        
        # 复制路径以避免修改原始数据
        best_routes = copy.deepcopy(routes)
        original_cost = self.calculate_total_cost(routes)
        best_cost = original_cost
        improved = False
        trials = 0
        
        if intra_route_only:
            # 只在路径内部应用2-opt
            for route_idx, route in enumerate(routes):
                if len(route) < 3:  # 至少需要3个客户才能应用2-opt
                    continue
                
                # 尝试所有可能的2-opt操作
                for i in range(len(route) - 1):
                    for j in range(i + 1, len(route)):
                        trials += 1
                        if trials > max_trials:
                            return best_routes, improved, original_cost - best_cost
                        
                        # 创建新的路径：翻转从i+1到j的部分
                        new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                        
                        # 检查时间窗可行性
                        if not self.check_time_window_feasibility(new_route):
                            continue
                        
                        # 创建新的路径集合
                        new_routes = copy.deepcopy(routes)
                        new_routes[route_idx] = new_route
                        
                        # 计算新成本
                        new_cost = self.calculate_total_cost(new_routes)
                        
                        # 如果有改进，更新最佳解
                        if new_cost < best_cost:
                            best_routes = new_routes
                            best_cost = new_cost
                            improved = True
                            
                            # 如果采用第一次改进策略，立即返回
                            if first_improvement:
                                return best_routes, True, original_cost - best_cost
        else:
            # 跨路径2-opt（更复杂，暂不实现）
            pass
        
        return best_routes, improved, original_cost - best_cost


class LocalSearchOperatorManager:
    """
    局部搜索算子管理器，用于管理和应用多个局部搜索算子
    """
    
    def __init__(self, instance, operators=None):
        """
        初始化局部搜索算子管理器
        
        Args:
            instance: 问题实例对象，包含距离矩阵、时间窗等信息
            operators: 初始算子字典，可选
        """
        self.instance = instance
        self.operators = operators or {}  # 存储算子的字典
    
    def add_operator(self, name: str, operator: LocalSearchOperator):
        """
        添加局部搜索算子
        
        Args:
            name: 算子名称
            operator: 局部搜索算子对象
        """
        self.operators[name] = operator
    
    def apply_operator(self, name: str, routes: List[List[int]], **kwargs) -> Tuple[List[List[int]], bool, float]:
        """
        应用指定的局部搜索算子
        
        Args:
            name: 算子名称
            routes: 路径列表
            **kwargs: 传递给算子的其他参数
            
        Returns:
            Tuple[List[List[int]], bool, float]: 
                - 优化后的路径列表
                - 是否有改进
                - 改进的成本差值
                
        Raises:
            KeyError: 如果指定的算子不存在
        """
        if name not in self.operators:
            raise KeyError(f"算子 '{name}' 不存在")
        
        return self.operators[name].apply(routes, **kwargs)
    
    def apply_all_operators(self, routes: List[List[int]], **kwargs) -> Tuple[List[List[int]], bool, float]:
        """
        依次应用所有局部搜索算子
        
        Args:
            routes: 路径列表
            **kwargs: 传递给算子的其他参数，可包含：
                - operator_order: 算子应用顺序列表
                - max_iterations: 最大迭代次数
                - improvement_threshold: 改进阈值，低于此值则停止迭代
                
        Returns:
            Tuple[List[List[int]], bool, float]: 
                - 优化后的路径列表
                - 是否有改进
                - 改进的成本差值
        """
        operator_order = kwargs.get('operator_order', list(self.operators.keys()))
        max_iterations = kwargs.get('max_iterations', 10)
        improvement_threshold = kwargs.get('improvement_threshold', 1e-6)
        
        best_routes = copy.deepcopy(routes)
        original_cost = route_cost(best_routes, self.instance.distance_matrix)
        best_cost = original_cost
        total_improved = False
        total_improvement = 0.0
        
        iteration = 0
        while iteration < max_iterations:
            iteration_improved = False
            iteration_improvement = 0.0
            
            for op_name in operator_order:
                current_routes, improved, improvement = self.apply_operator(op_name, best_routes, **kwargs)
                
                if improved:
                    best_routes = current_routes
                    best_cost -= improvement
                    iteration_improved = True
                    iteration_improvement += improvement
            
            if not iteration_improved or iteration_improvement < improvement_threshold:
                break
                
            total_improved = True
            total_improvement += iteration_improvement
            iteration += 1
        
        return best_routes, total_improved, total_improvement
    
    def create_standard_operators(self):
        """
        创建标准的局部搜索算子集合
        
        Returns:
            self: 返回自身，支持链式调用
        """
        if not self.instance:
            raise ValueError("需要提供问题实例才能创建标准算子")
            
        # 创建标准算子
        self.add_operator("relocate", RelocateOperator(self.instance))
        self.add_operator("swap", SwapOperator(self.instance))
        self.add_operator("or_opt", OrOptOperator(self.instance))
        self.add_operator("cross_exchange", CrossExchangeOperator(self.instance))
        self.add_operator("two_opt", TwoOptOperator(self.instance))
        
        return self
        初始化局部搜索算子管理器
        
        Args:
            instance: VRPTW实例对象
            operators: 局部搜索算子列表，如果为None则使用默认算子
        """
        self.instance = instance
        self.operators = operators if operators is not None else []
        
    def add_operator(self, operator: LocalSearchOperator):
        """
        添加局部搜索算子
        
        Args:
            operator: 局部搜索算子
        """
        self.operators.append(operator)
        
    def apply_operators(self, routes: List[List[int]], strategy='first_improvement', 
                        max_iterations=100, verbose=False) -> Tuple[List[List[int]], float]:
        """
        应用所有局部搜索算子到给定的路径集合
        
        Args:
            routes: 路径列表，每个路径是客户索引列表
            strategy: 应用策略，可选值：'first_improvement'（第一次改进）, 'best_improvement'（最佳改进）
            max_iterations: 最大迭代次数
            verbose: 是否输出详细信息
            
        Returns:
            Tuple[List[List[int]], float]: 
                - 优化后的路径列表
                - 总改进的成本
        """
        current_routes = copy.deepcopy(routes)
        initial_cost = sum(route_cost(route, self.instance.distance_matrix) for route in current_routes)
        current_cost = initial_cost
        
        if verbose:
            print(f"初始成本: {initial_cost:.2f}")
        
        iteration = 0
        improved = True
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            if verbose:
                print(f"迭代 {iteration}:")
            
            # 应用每个算子
            for operator in self.operators:
                operator_name = operator.__class__.__name__
                
                if verbose:
                    print(f"  应用算子: {operator_name}")
                
                new_routes, operator_improved, cost_change = operator.apply(current_routes)
                
                # 如果有改进，更新当前解
                if operator_improved:
                    current_routes = new_routes
                    current_cost -= cost_change
                    improved = True
                    
                    if verbose:
                        print(f"    改进: {cost_change:.2f}, 当前成本: {current_cost:.2f}")
                    
                    # 如果策略是第一次改进，则立即返回
                    if strategy == 'first_improvement':
                        break
            
            if verbose and not improved:
                print("  没有找到改进")
        
        total_improvement = initial_cost - current_cost
        
        if verbose:
            print(f"总迭代次数: {iteration}")
            print(f"总改进: {total_improvement:.2f}")
            print(f"最终成本: {current_cost:.2f}")
        
        return current_routes, total_improvement


# 辅助函数
def route_cost(route, dist_mat):
    """
    计算单条路由的总距离：depot → route → depot
    
    Args:
        route: 客户索引列表（0-based，对应 dist_mat 中的行/列要 +1）
        dist_mat: 距离矩阵，大小 (N+1)×(N+1)，dist_mat[0] 是 depot
        
    Returns:
        float: 总距离
    """
    cost = 0.0
    prev = 0  # 从 depot（索引 0）出发
    for cust in route:
        # 客户在 dist_mat 中索引为 cust+1
        cost += dist_mat[prev][cust + 1]
        prev = cust + 1
    # 返回 depot
    cost += dist_mat[prev][0]
    return cost


def decode_chromosome_to_routes(chromosome, decoder):
    """
    将染色体解码为路径列表
    
    Args:
        chromosome: 染色体（客户索引列表）
        decoder: 解码器对象
        
    Returns:
        List[List[int]]: 路径列表，每个路径是客户索引列表
    """
    solution = decoder.decode_solution(chromosome)
    routes = []
    
    for route_info in solution['routes']:
        routes.append(route_info['customers'])
    
    return routes


def encode_routes_to_chromosome(routes):
    """
    将路径列表编码为染色体
    
    Args:
        routes: 路径列表，每个路径是客户索引列表
        
    Returns:
        List[int]: 染色体（客户索引列表）
    """
    chromosome = []
    for route in routes:
        chromosome.extend(route)
    return chromosome