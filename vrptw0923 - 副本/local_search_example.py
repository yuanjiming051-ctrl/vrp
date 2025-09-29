#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VRPTW局部搜索算子使用示例
"""

import numpy as np
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass

from local_search_operators import (
    LocalSearchOperator, 
    RelocateOperator, 
    SwapOperator, 
    OrOptOperator, 
    CrossExchangeOperator, 
    TwoOptOperator, 
    LocalSearchOperatorManager
)


@dataclass
class VRPTWInstance:
    """VRPTW问题实例"""
    distance_matrix: np.ndarray
    time_windows: List[Tuple[float, float]]
    demands: List[float]
    capacity: float
    
    @classmethod
    def create_random_instance(cls, num_customers=50, seed=42):
        """创建随机问题实例"""
        np.random.seed(seed)
        
        # 生成随机客户位置
        locations = np.random.rand(num_customers + 1, 2) * 100  # +1 表示包含仓库
        
        # 计算距离矩阵
        distance_matrix = np.zeros((num_customers + 1, num_customers + 1))
        for i in range(num_customers + 1):
            for j in range(num_customers + 1):
                if i != j:
                    distance_matrix[i, j] = np.sqrt(
                        (locations[i, 0] - locations[j, 0]) ** 2 + 
                        (locations[i, 1] - locations[j, 1]) ** 2
                    )
        
        # 生成随机时间窗
        time_windows = [(0, 1000)]  # 仓库时间窗
        for _ in range(num_customers):
            earliest = np.random.uniform(0, 500)
            latest = earliest + np.random.uniform(50, 200)
            time_windows.append((earliest, latest))
        
        # 生成随机需求
        demands = [0]  # 仓库需求为0
        for _ in range(num_customers):
            demands.append(np.random.uniform(1, 20))
        
        # 设置车辆容量
        capacity = 100
        
        return cls(
            distance_matrix=distance_matrix,
            time_windows=time_windows,
            demands=demands,
            capacity=capacity
        )


def create_initial_solution(instance: VRPTWInstance, num_vehicles=10) -> List[List[int]]:
    """创建初始解决方案（简单贪心策略）"""
    num_customers = len(instance.demands) - 1  # 减去仓库
    customers = list(range(1, num_customers + 1))
    
    # 按照时间窗早晚排序客户
    customers.sort(key=lambda c: instance.time_windows[c][0])
    
    routes = []
    for _ in range(num_vehicles):
        routes.append([])
    
    # 简单分配客户到路径
    for i, customer in enumerate(customers):
        route_idx = i % num_vehicles
        routes[route_idx].append(customer)
    
    # 移除空路径
    routes = [r for r in routes if r]
    
    return routes


def print_solution(routes: List[List[int]], instance: VRPTWInstance):
    """打印解决方案"""
    total_cost = 0
    print("\n解决方案:")
    for i, route in enumerate(routes):
        if not route:
            continue
            
        route_cost = 0
        route_load = 0
        
        # 从仓库到第一个客户
        route_cost += instance.distance_matrix[0, route[0]]
        
        # 客户之间
        for j in range(len(route) - 1):
            route_cost += instance.distance_matrix[route[j], route[j + 1]]
            route_load += instance.demands[route[j]]
        
        # 最后一个客户到仓库
        route_cost += instance.distance_matrix[route[-1], 0]
        route_load += instance.demands[route[-1]]
        
        print(f"路径 {i+1}: {' -> '.join(['0'] + [str(c) for c in route] + ['0'])} | 成本: {route_cost:.2f} | 负载: {route_load:.2f}")
        total_cost += route_cost
    
    print(f"\n总成本: {total_cost:.2f}")
    print(f"路径数量: {len([r for r in routes if r])}")
    return total_cost


def main():
    """主函数"""
    print("创建随机VRPTW问题实例...")
    instance = VRPTWInstance.create_random_instance(num_customers=30, seed=42)
    
    print("生成初始解决方案...")
    initial_routes = create_initial_solution(instance, num_vehicles=5)
    initial_cost = print_solution(initial_routes, instance)
    
    print("\n创建局部搜索算子管理器...")
    manager = LocalSearchOperatorManager(instance)
    
    # 添加所有算子
    print("添加局部搜索算子...")
    manager.add_operator("relocate", RelocateOperator(instance))
    manager.add_operator("swap", SwapOperator(instance))
    manager.add_operator("or_opt", OrOptOperator(instance, segment_length=2))
    manager.add_operator("cross_exchange", CrossExchangeOperator(instance, max_segment_length=2))
    manager.add_operator("two_opt", TwoOptOperator(instance))
    
    # 单独测试每个算子
    print("\n单独测试每个算子:")
    routes = initial_routes.copy()
    for op_name in ["relocate", "swap", "or_opt", "cross_exchange", "two_opt"]:
        print(f"\n应用 {op_name} 算子...")
        start_time = time.time()
        new_routes, improved, improvement = manager.apply_operator(
            op_name, routes, max_trials=1000, first_improvement=False
        )
        end_time = time.time()
        
        if improved:
            routes = new_routes
            print(f"改进成功! 成本减少: {improvement:.2f}, 耗时: {end_time - start_time:.2f}秒")
        else:
            print(f"未找到改进, 耗时: {end_time - start_time:.2f}秒")
    
    print("\n单独算子优化后的解决方案:")
    single_op_cost = print_solution(routes, instance)
    
    # 应用所有算子
    print("\n应用所有算子进行优化...")
    start_time = time.time()
    final_routes, improved, total_improvement = manager.apply_all_operators(
        initial_routes,
        operator_order=["relocate", "swap", "or_opt", "cross_exchange", "two_opt"],
        max_iterations=5,
        max_trials=500,
        first_improvement=True
    )
    end_time = time.time()
    
    print(f"\n所有算子优化后的解决方案 (耗时: {end_time - start_time:.2f}秒):")
    final_cost = print_solution(final_routes, instance)
    
    # 打印总结
    print("\n优化总结:")
    print(f"初始解成本: {initial_cost:.2f}")
    print(f"单独算子优化后成本: {single_op_cost:.2f} (改进: {initial_cost - single_op_cost:.2f}, {(initial_cost - single_op_cost) / initial_cost * 100:.2f}%)")
    print(f"所有算子优化后成本: {final_cost:.2f} (改进: {initial_cost - final_cost:.2f}, {(initial_cost - final_cost) / initial_cost * 100:.2f}%)")


if __name__ == "__main__":
    main()