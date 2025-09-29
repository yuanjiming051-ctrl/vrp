# File: decoder.py
# 高性能VRPTW解码器 - 支持多种解码策略

import numpy as np
from numba import njit
import random
from typing import List, Dict, Tuple, Optional

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@njit
def _fast_decode_jit(chromosome, distance_matrix, capacity, demands, ready_times, due_times, service_times):
    """
    使用Numba JIT加速的核心解码函数
    返回: (vehicle_count, total_distance, route_starts, route_ends, feasible)
    """
    n_customers = len(chromosome)
    routes_start = np.zeros(n_customers, dtype=np.int32)  # 每条路径的起始索引
    routes_end = np.zeros(n_customers, dtype=np.int32)    # 每条路径的结束索引
    
    vehicle_count = 0
    total_distance = 0.0
    current_load = 0.0
    current_time = 0.0
    current_position = 0  # depot
    feasible = True
    route_started = False
    
    for i in range(n_customers):
        customer_idx = chromosome[i]
        demand = demands[customer_idx]
        ready_time = ready_times[customer_idx]
        due_time = due_times[customer_idx]
        service_time = service_times[customer_idx]
        
        # 计算到达时间
        travel_time = distance_matrix[current_position, customer_idx + 1]
        arrival_time = current_time + travel_time
        
        # 计算开始服务时间（考虑等待时间）
        start_service_time = max(arrival_time, ready_time)
        
        # 检查容量和时间窗约束
        if (current_load + demand > capacity or start_service_time > due_time):
            # 需要新车辆
            if route_started:  # 完成当前路径
                # 回到depot
                total_distance += distance_matrix[current_position, 0]
                routes_end[vehicle_count - 1] = i - 1
            
            # 开始新路径
            routes_start[vehicle_count] = i
            vehicle_count += 1
            route_started = True
            
            # 重新计算从depot到当前客户
            current_load = demand
            current_position = customer_idx + 1
            travel_time = distance_matrix[0, customer_idx + 1]
            arrival_time = travel_time
            start_service_time = max(arrival_time, ready_time)
            
            # 检查新路径的可行性
            if start_service_time > due_time:
                feasible = False
            
            current_time = start_service_time + service_time
            total_distance += travel_time
        else:
            # 添加到当前路径
            if not route_started:
                # 第一个客户，开始第一条路径
                routes_start[vehicle_count] = i
                vehicle_count += 1
                route_started = True
            
            current_load += demand
            current_position = customer_idx + 1
            current_time = start_service_time + service_time
            total_distance += travel_time
    
    # 处理最后一条路径
    if route_started:
        total_distance += distance_matrix[current_position, 0]
        routes_end[vehicle_count - 1] = n_customers - 1
    
    return vehicle_count, total_distance, routes_start, routes_end, feasible


@njit
def _calculate_route_metrics_jit(route_customers, distance_matrix, ready_times, due_times, service_times):
    """
    使用JIT计算单条路径的详细指标
    返回: (distance, total_time, travel_time, service_time, wait_time, feasible)
    """
    if len(route_customers) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, True
    
    distance = 0.0
    total_time = 0.0
    travel_time = 0.0
    total_service_time = 0.0
    wait_time = 0.0
    current_time = 0.0
    current_pos = 0  # depot
    feasible = True
    
    for i in range(len(route_customers)):
        customer_idx = route_customers[i]
        
        # 计算距离和旅行时间
        travel_dist = distance_matrix[current_pos, customer_idx + 1]
        distance += travel_dist
        travel_time += travel_dist
        
        # 计算到达和服务时间
        arrival_time = current_time + travel_dist
        ready_time = ready_times[customer_idx]
        due_time = due_times[customer_idx]
        service_time = service_times[customer_idx]
        
        # 检查时间窗约束
        if arrival_time > due_time:
            feasible = False
        
        # 计算等待时间
        if arrival_time < ready_time:
            wait_time += ready_time - arrival_time
            current_time = ready_time + service_time
        else:
            current_time = arrival_time + service_time
        
        total_service_time += service_time
        current_pos = customer_idx + 1
    
    # 返回depot
    return_dist = distance_matrix[current_pos, 0]
    distance += return_dist
    travel_time += return_dist
    total_time = current_time + return_dist
    
    return distance, total_time, travel_time, total_service_time, wait_time, feasible


class VRPTWDecoder:
    """
    高性能VRPTW解码器，支持多种解码策略
    """
    
    def __init__(self, instance):
        self.vehicle_capacity = instance.vehicle_info['capacity']
        self.customers = instance.ordinary_customers
        self.distance_matrix = instance.distance_matrix
        self.warehouse_idx = 0
        
        # 预处理客户数据为NumPy数组以提高性能
        n_customers = len(self.customers)
        self.demands = np.array([c['demand'] for c in self.customers], dtype=np.float64)
        self.ready_times = np.array([c['ready_time'] for c in self.customers], dtype=np.float64)
        self.due_times = np.array([c['due_date'] for c in self.customers], dtype=np.float64)
        self.service_times = np.array([c['service_time'] for c in self.customers], dtype=np.float64)
    
    def decode_solution(self, chromosome: List[int], strategy: str = 'fast', space: str = 'full', ds=None) -> Dict:
        """
        解码染色体为VRPTW解决方案
        
        Args:
            chromosome: 客户访问顺序
            strategy: 解码策略 ('fast', 'detailed', 'hybrid')
            space: 空间类型 ('full', 'deep')
            ds: Deep空间数据，当space='deep'时需要提供
        
        Returns:
            包含车辆数、总距离、路径等信息的字典
        """
        # 处理Deep空间的染色体
        if space == 'deep':
            if ds is None:
                raise ValueError("解码Deep空间染色体需要提供ds参数")
            
            if 'VC_new' not in ds:
                raise ValueError("ds参数中缺少VC_new字段")
                
            # print(f"[Deep解码] Deep空间染色体长度: {len(chromosome)}")
            
            # 将Deep空间的染色体映射到Full空间
            full_chromosome = []
            for deep_idx in chromosome:
                if isinstance(deep_idx, (int, np.integer)) and 0 <= deep_idx < len(ds['VC_new']):
                    full_chromosome.extend(ds['VC_new'][deep_idx])
                else:
                    print(f"警告：Deep空间索引 {deep_idx} (类型: {type(deep_idx)}) 超出范围 [0, {len(ds['VC_new'])-1}]")
            
            # print(f"[Deep解码] 映射到Full空间后染色体长度: {len(full_chromosome)}")
            
            # 使用映射后的Full空间染色体进行解码
            chromosome = full_chromosome
        
        # 使用选定的策略进行解码
        if strategy == 'fast':
            result = self._decode_fast(chromosome)
        elif strategy == 'detailed':
            result = self._decode_detailed(chromosome)
        elif strategy == 'hybrid':
            result = self._decode_hybrid(chromosome)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # 如果是Deep空间解码，输出路径数量信息
        if space == 'deep':
            pass  # print(f"[Deep解码] 解码后路径数量: {result['vehicle_count']}, 总成本: {result['total_distance']:.2f}")
        
        return result
    
    def _decode_fast(self, chromosome: List[int]) -> Dict:
        """
        快速解码策略 - 使用JIT加速，只计算基本指标
        """
        # 边界检查：过滤无效索引
        valid_chromosome = [idx for idx in chromosome if 0 <= idx < len(self.customers)]
        if len(valid_chromosome) != len(chromosome):
            print(f"警告：染色体中有 {len(chromosome) - len(valid_chromosome)} 个无效索引被过滤")
        
        chromosome_array = np.array(valid_chromosome, dtype=np.int32)
        
        vehicle_count, total_distance, route_starts, route_ends, feasible = _fast_decode_jit(
            chromosome_array, self.distance_matrix, self.vehicle_capacity,
            self.demands, self.ready_times, self.due_times, self.service_times
        )
        
        # 构建路径信息
        routes = []
        for i in range(vehicle_count):
            start_idx = route_starts[i]
            end_idx = route_ends[i]
            customers = chromosome[start_idx:end_idx + 1]
            routes.append({
                'customers': customers,
                'distance': 0.0  # 快速模式不计算单独路径距离
            })
        
        return {
            'vehicle_count': vehicle_count,
            'total_distance': total_distance,
            'routes': routes,
            'feasible': feasible,
            'strategy': 'fast'
        }
    
    def _decode_detailed(self, chromosome: List[int]) -> Dict:
        """
        详细解码策略 - 计算完整的路径信息和指标
        """
        routes = []
        current_route = []
        current_load = 0.0
        current_time = 0.0
        
        for customer_idx in chromosome:
            # 边界检查：确保索引在有效范围内
            if customer_idx < 0 or customer_idx >= len(self.customers):
                print(f"警告：客户索引 {customer_idx} 超出范围 [0, {len(self.customers)-1}]，跳过该客户")
                continue
                
            customer = self.customers[customer_idx]
            demand = customer['demand']
            ready_time = customer['ready_time']
            due_time = customer['due_date']
            service_time = customer['service_time']
            
            # 计算到达时间
            if not current_route:
                travel_time = self.distance_matrix[0, customer_idx + 1]
            else:
                prev_customer = current_route[-1]
                travel_time = self.distance_matrix[prev_customer + 1, customer_idx + 1]
            
            arrival_time = current_time + travel_time
            
            # 严格检查约束：载重和时间窗
            if (current_load + demand > self.vehicle_capacity or 
                arrival_time > due_time):
                
                # 完成当前路径
                if current_route:
                    route_info = self._calculate_route_info(current_route)
                    routes.append(route_info)
                
                # 开始新路径
                current_route = [customer_idx]
                current_load = demand
                travel_time = self.distance_matrix[0, customer_idx + 1]
                arrival_time = travel_time
                current_time = max(arrival_time, ready_time) + service_time
                
                # 检查单个客户是否超载
                if demand > self.vehicle_capacity:
                    print(f"错误：客户{customer_idx}需求{demand}超过车辆容量{self.vehicle_capacity}")
            else:
                # 添加到当前路径
                current_route.append(customer_idx)
                current_load += demand
                current_time = max(arrival_time, ready_time) + service_time
        
        # 处理最后一条路径
        if current_route:
            route_info = self._calculate_route_info(current_route)
            routes.append(route_info)
        
        total_distance = sum(route['distance'] for route in routes)
        feasible = all(route['feasible'] for route in routes)
        
        return {
            'vehicle_count': len(routes),
            'total_distance': total_distance,
            'routes': routes,
            'feasible': feasible,
            'strategy': 'detailed'
        }
    
    def _decode_hybrid(self, chromosome: List[int]) -> Dict:
        """
        混合解码策略 - 先快速计算，再按需计算详细信息
        """
        # 先用快速策略获取基本结果
        fast_result = self._decode_fast(chromosome)
        
        # 如果需要详细信息，再计算路径详情
        detailed_routes = []
        for route in fast_result['routes']:
            if route['customers']:
                route_info = self._calculate_route_info(route['customers'])
                detailed_routes.append(route_info)
        
        fast_result['routes'] = detailed_routes
        fast_result['strategy'] = 'hybrid'
        
        return fast_result
    
    def _calculate_route_info(self, route_customers: List[int]) -> Dict:
        """
        计算单条路径的详细信息
        """
        if not route_customers:
            return {
                'customers': [],
                'distance': 0.0,
                'total_time': 0.0,
                'travel_time': 0.0,
                'service_time': 0.0,
                'wait_time': 0.0,
                'load': 0.0,
                'feasible': True
            }
        
        route_array = np.array(route_customers, dtype=np.int32)
        distance, total_time, travel_time, service_time, wait_time, feasible = _calculate_route_metrics_jit(
            route_array, self.distance_matrix, self.ready_times, self.due_times, self.service_times
        )
        
        load = sum(self.demands[idx] for idx in route_customers)
        
        # 检查载重约束
        if load > self.vehicle_capacity:
            feasible = False
            # 警告：发现超载路径
            print(f"警告：路径超载 {load}/{self.vehicle_capacity} ({load/self.vehicle_capacity*100:.1f}%)")
        
        return {
            'customers': route_customers,
            'distance': distance,
            'total_time': total_time,
            'travel_time': travel_time,
            'service_time': service_time,
            'wait_time': wait_time,
            'load': load,
            'feasible': feasible
        }
    
    def _calculate_route_cost(self, route: List[int]) -> float:
        """
        计算路径成本（距离）
        """
        if not route:
            return 0.0
        
        cost = 0.0
        # depot → first customer
        cost += self.distance_matrix[0, route[0] + 1]
        
        # between customers
        for i in range(len(route) - 1):
            cost += self.distance_matrix[route[i] + 1, route[i + 1] + 1]
        
        # last customer → depot
        cost += self.distance_matrix[route[-1] + 1, 0]
        
        return cost
    
    def visualize_routes(self, routes: List[Dict], instance, title: str = "VRPTW Routes"):
        """
        可视化路径
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib未安装，无法可视化路径")
            return
            
        plt.figure(figsize=(12, 8))
        
        # 绘制depot
        depot = instance.warehouse
        plt.scatter(depot['xcoord'], depot['ycoord'], c='red', marker='s', s=200, label='Depot')
        
        # 绘制客户
        customer_x = [c['xcoord'] for c in instance.ordinary_customers]
        customer_y = [c['ycoord'] for c in instance.ordinary_customers]
        plt.scatter(customer_x, customer_y, c='blue', s=50, label='Customers')
        
        # 标注客户编号
        for i, customer in enumerate(instance.ordinary_customers):
            plt.annotate(str(customer['cust_no']),
                        (customer['xcoord'], customer['ycoord']),
                        xytext=(5, 5), textcoords='offset points')
        
        # 绘制路径
        colors = plt.cm.rainbow(np.linspace(0, 1, len(routes)))
        for i, route in enumerate(routes):
            color = colors[i]
            
            # 构建路径点
            points = [(depot['xcoord'], depot['ycoord'])]
            for customer_idx in route['customers']:
                customer = instance.ordinary_customers[customer_idx]
                points.append((customer['xcoord'], customer['ycoord']))
            points.append((depot['xcoord'], depot['ycoord']))
            
            # 绘制路径线
            for j in range(len(points) - 1):
                x_coords = [points[j][0], points[j + 1][0]]
                y_coords = [points[j][1], points[j + 1][1]]
                plt.plot(x_coords, y_coords, c=color, linewidth=1.5, alpha=0.7)
            
            # 标注车辆编号
            plt.annotate(f'V{i + 1}', (depot['xcoord'], depot['ycoord']), 
                        color=color, fontweight='bold', fontsize=10)
        
        plt.title(title)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def visualize_routes(self, routes, instance, title="VRPTW Routes"):
        """可视化路径（保持原有功能不变）"""
        plt.figure(figsize=(12, 8))
        depot = instance.warehouse
        plt.scatter(depot['xcoord'], depot['ycoord'], c='red', marker='s', s=200, label='Depot')

        xs = [c['xcoord'] for c in instance.ordinary_customers]
        ys = [c['ycoord'] for c in instance.ordinary_customers]
        plt.scatter(xs, ys, c='blue', s=50, label='Customers')

        for i, cust in enumerate(instance.ordinary_customers):
            plt.annotate(str(cust['cust_no']), (cust['xcoord'], cust['ycoord']), xytext=(5, 5),
                         textcoords='offset points')

        cmap = plt.cm.rainbow(np.linspace(0, 1, len(routes)))
        for i, r in enumerate(routes):
            color = cmap[i]
            pts = [(depot['xcoord'], depot['ycoord'])]
            for idx in r['customers']:
                c = instance.ordinary_customers[idx]
                pts.append((c['xcoord'], c['ycoord']))
            pts.append((depot['xcoord'], depot['ycoord']))

            for j in range(len(pts) - 1):
                x0, y0 = pts[j]
                x1, y1 = pts[j + 1]
                plt.plot([x0, x1], [y0, y1], c=color, lw=1.5, alpha=0.7)
                if j < len(pts) - 2:
                    midx = (x0 + x1) / 2
                    midy = (y0 + y1) / 2
                    dx = (x1 - x0) / 10
                    dy = (y1 - y0) / 10
                    plt.arrow(midx, midy, dx, dy, head_width=3, head_length=4, fc=color, ec=color, alpha=0.7)

            plt.annotate(f'V{i + 1}', (pts[0][0] + random.randint(-10, 10), pts[0][1] + random.randint(-10, 10)),
                         color=color, fontweight='bold')

        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(linestyle='--', alpha=0.6)
        plt.tight_layout()
#        plt.show()'''
