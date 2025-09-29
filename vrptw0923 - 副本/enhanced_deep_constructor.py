#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的Deep空间构造器
为聚类节点添加开始位置和结束位置信息，提供更丰富的空间表示
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Any
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
import random


class EnhancedDeepNode:
    """增强的Deep空间节点，包含开始和结束位置信息"""
    
    def __init__(self, node_id: int, customers: List[int], instance):
        self.node_id = node_id
        self.customers = customers  # 包含的原始客户列表
        self.instance = instance
        
        # 计算聚类的基本信息
        self._calculate_cluster_info()
        
        # 计算开始和结束位置
        self._calculate_positions()
        
        # 计算时间窗信息
        self._calculate_time_windows()
        
        # 计算服务信息
        self._calculate_service_info()
    
    def _calculate_cluster_info(self):
        """计算聚类的基本信息"""
        if not self.customers:
            raise ValueError("聚类不能为空")
        
        # 计算聚类中心（重心）
        x_coords = [self.instance.customers[c]['x'] for c in self.customers]
        y_coords = [self.instance.customers[c]['y'] for c in self.customers]
        
        self.center_x = sum(x_coords) / len(x_coords)
        self.center_y = sum(y_coords) / len(y_coords)
        
        # 计算聚类半径（最远客户到中心的距离）
        max_dist = 0
        for c in self.customers:
            dist = math.sqrt((self.instance.customers[c]['x'] - self.center_x)**2 + 
                           (self.instance.customers[c]['y'] - self.center_y)**2)
            max_dist = max(max_dist, dist)
        self.radius = max_dist
        
        # 计算总需求
        self.total_demand = sum(self.instance.customers[c]['demand'] for c in self.customers)
    
    def _calculate_positions(self):
        """计算开始位置和结束位置"""
        if len(self.customers) == 1:
            # 单客户聚类：开始和结束位置相同
            c = self.customers[0]
            self.start_x = self.instance.customers[c]['x']
            self.start_y = self.instance.customers[c]['y']
            self.end_x = self.start_x
            self.end_y = self.start_y
        else:
            # 多客户聚类：计算最优的开始和结束位置
            self._calculate_optimal_start_end_positions()
    
    def _calculate_optimal_start_end_positions(self):
        """计算最优的开始和结束位置"""
        # 方法1：基于时间窗的最早和最晚客户
        earliest_customer = min(self.customers, 
                              key=lambda c: self.instance.customers[c]['ready_time'])
        latest_customer = max(self.customers, 
                            key=lambda c: self.instance.customers[c]['due_time'])
        
        # 开始位置：最早时间窗客户的位置
        self.start_x = self.instance.customers[earliest_customer]['x']
        self.start_y = self.instance.customers[earliest_customer]['y']
        
        # 结束位置：最晚时间窗客户的位置
        self.end_x = self.instance.customers[latest_customer]['x']
        self.end_y = self.instance.customers[latest_customer]['y']
        
        # 方法2：基于几何位置的边界点（备选方案）
        # 计算聚类的边界点
        x_coords = [self.instance.customers[c]['x'] for c in self.customers]
        y_coords = [self.instance.customers[c]['y'] for c in self.customers]
        
        # 找到最左和最右的点作为备选
        leftmost_idx = np.argmin(x_coords)
        rightmost_idx = np.argmax(x_coords)
        
        self.alt_start_x = x_coords[leftmost_idx]
        self.alt_start_y = y_coords[leftmost_idx]
        self.alt_end_x = x_coords[rightmost_idx]
        self.alt_end_y = y_coords[rightmost_idx]
    
    def _calculate_time_windows(self):
        """计算聚类的时间窗"""
        # 聚类的最早开始时间：所有客户中最早的ready_time
        self.ready_time = min(self.instance.customers[c]['ready_time'] for c in self.customers)
        
        # 聚类的最晚结束时间：所有客户中最晚的due_time
        self.due_time = max(self.instance.customers[c]['due_time'] for c in self.customers)
        
        # 计算聚类内部的服务时间（遍历所有客户所需的最短时间）
        self.internal_service_time = self._calculate_internal_service_time()
    
    def _calculate_internal_service_time(self):
        """计算聚类内部服务时间（TSP近似）"""
        if len(self.customers) <= 1:
            return self.instance.customers[self.customers[0]]['service_time'] if self.customers else 0
        
        # 使用最近邻启发式计算TSP近似
        unvisited = set(self.customers)
        current = self.customers[0]
        unvisited.remove(current)
        
        total_time = self.instance.customers[current]['service_time']
        
        while unvisited:
            # 找到最近的未访问客户
            nearest = min(unvisited, 
                         key=lambda c: self._distance(current, c))
            
            # 添加旅行时间和服务时间
            total_time += self._distance(current, nearest)
            total_time += self.instance.customers[nearest]['service_time']
            
            current = nearest
            unvisited.remove(current)
        
        return total_time
    
    def _distance(self, c1: int, c2: int) -> float:
        """计算两个客户之间的距离"""
        x1, y1 = self.instance.customers[c1]['x'], self.instance.customers[c1]['y']
        x2, y2 = self.instance.customers[c2]['x'], self.instance.customers[c2]['y']
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _calculate_service_info(self):
        """计算服务相关信息"""
        # 平均服务时间
        self.avg_service_time = sum(self.instance.customers[c]['service_time'] 
                                  for c in self.customers) / len(self.customers)
        
        # 客户数量
        self.customer_count = len(self.customers)
        
        # 紧急度（基于时间窗紧张程度）
        time_window_spans = [self.instance.customers[c]['due_time'] - self.instance.customers[c]['ready_time'] 
                           for c in self.customers]
        self.urgency = 1.0 / (1.0 + np.mean(time_window_spans))  # 时间窗越小，紧急度越高
    
    def get_position_info(self) -> Dict[str, Any]:
        """获取位置信息"""
        return {
            'center': (self.center_x, self.center_y),
            'start_position': (self.start_x, self.start_y),
            'end_position': (self.end_x, self.end_y),
            'alt_start_position': (self.alt_start_x, self.alt_start_y),
            'alt_end_position': (self.alt_end_x, self.alt_end_y),
            'radius': self.radius
        }
    
    def get_time_info(self) -> Dict[str, Any]:
        """获取时间信息"""
        return {
            'ready_time': self.ready_time,
            'due_time': self.due_time,
            'internal_service_time': self.internal_service_time,
            'avg_service_time': self.avg_service_time,
            'urgency': self.urgency
        }
    
    def get_service_info(self) -> Dict[str, Any]:
        """获取服务信息"""
        return {
            'total_demand': self.total_demand,
            'customer_count': self.customer_count,
            'customers': self.customers.copy()
        }
    
    def distance_to_node(self, other_node: 'EnhancedDeepNode', mode: str = 'end_to_start') -> float:
        """计算到另一个节点的距离"""
        if mode == 'end_to_start':
            # 从当前节点的结束位置到目标节点的开始位置
            return math.sqrt((other_node.start_x - self.end_x)**2 + 
                           (other_node.start_y - self.end_y)**2)
        elif mode == 'center_to_center':
            # 中心到中心的距离
            return math.sqrt((other_node.center_x - self.center_x)**2 + 
                           (other_node.center_y - self.center_y)**2)
        elif mode == 'start_to_start':
            # 开始位置到开始位置
            return math.sqrt((other_node.start_x - self.start_x)**2 + 
                           (other_node.start_y - self.start_y)**2)
        else:
            raise ValueError(f"未知的距离模式: {mode}")
    
    def __repr__(self):
        return f"EnhancedDeepNode(id={self.node_id}, customers={len(self.customers)}, demand={self.total_demand})"


class EnhancedDeepConstructor:
    """增强的Deep空间构造器"""
    
    def __init__(self, instance, cluster_method: str = 'kmeans', 
                 position_strategy: str = 'time_based'):
        self.instance = instance
        self.cluster_method = cluster_method
        self.position_strategy = position_strategy
        self.deep_nodes = []
        self.node_mapping = {}  # 原始客户到Deep节点的映射
    
    def construct_deep_space(self, target_clusters: int = None) -> List[EnhancedDeepNode]:
        """构造增强的Deep空间"""
        if target_clusters is None:
            # 自动确定聚类数量（客户数的1/3到1/2之间）
            n_customers = len(self.instance.customers)
            target_clusters = max(3, min(n_customers // 3, n_customers // 2))
        
        # 执行聚类
        clusters = self._perform_clustering(target_clusters)
        
        # 创建增强的Deep节点
        self.deep_nodes = []
        self.node_mapping = {}
        
        for i, cluster in enumerate(clusters):
            if cluster:  # 确保聚类不为空
                deep_node = EnhancedDeepNode(i, cluster, self.instance)
                self.deep_nodes.append(deep_node)
                
                # 更新映射
                for customer in cluster:
                    self.node_mapping[customer] = i
        
        return self.deep_nodes
    
    def _perform_clustering(self, n_clusters: int) -> List[List[int]]:
        """执行聚类算法"""
        customers = list(self.instance.customers.keys())
        
        if self.cluster_method == 'kmeans':
            return self._kmeans_clustering(customers, n_clusters)
        elif self.cluster_method == 'time_spatial':
            return self._time_spatial_clustering(customers, n_clusters)
        elif self.cluster_method == 'demand_based':
            return self._demand_based_clustering(customers, n_clusters)
        else:
            raise ValueError(f"未知的聚类方法: {self.cluster_method}")
    
    def _kmeans_clustering(self, customers: List[int], n_clusters: int) -> List[List[int]]:
        """基于K-means的聚类"""
        # 准备特征矩阵（位置 + 时间窗 + 需求）
        features = []
        for c in customers:
            customer_data = self.instance.customers[c]
            features.append([
                customer_data['x'],
                customer_data['y'],
                customer_data['ready_time'] / 100,  # 归一化时间
                customer_data['due_time'] / 100,
                customer_data['demand'] / 10  # 归一化需求
            ])
        
        features = np.array(features)
        
        # 执行K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        # 组织聚类结果
        clusters = [[] for _ in range(n_clusters)]
        for i, label in enumerate(labels):
            clusters[label].append(customers[i])
        
        # 移除空聚类
        return [cluster for cluster in clusters if cluster]
    
    def _time_spatial_clustering(self, customers: List[int], n_clusters: int) -> List[List[int]]:
        """基于时间-空间的聚类"""
        # 使用时间窗和空间位置的加权组合
        features = []
        for c in customers:
            customer_data = self.instance.customers[c]
            # 时间窗中心点
            time_center = (customer_data['ready_time'] + customer_data['due_time']) / 2
            features.append([
                customer_data['x'] * 0.7,  # 空间权重
                customer_data['y'] * 0.7,
                time_center * 0.3 / 100,   # 时间权重（归一化）
                customer_data['demand'] * 0.1 / 10  # 需求权重
            ])
        
        features = np.array(features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        clusters = [[] for _ in range(n_clusters)]
        for i, label in enumerate(labels):
            clusters[label].append(customers[i])
        
        return [cluster for cluster in clusters if cluster]
    
    def _demand_based_clustering(self, customers: List[int], n_clusters: int) -> List[List[int]]:
        """基于需求平衡的聚类"""
        # 先按空间位置聚类，然后调整以平衡需求
        spatial_features = []
        for c in customers:
            customer_data = self.instance.customers[c]
            spatial_features.append([customer_data['x'], customer_data['y']])
        
        spatial_features = np.array(spatial_features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(spatial_features)
        
        # 初始聚类
        clusters = [[] for _ in range(n_clusters)]
        for i, label in enumerate(labels):
            clusters[label].append(customers[i])
        
        # 需求平衡调整
        self._balance_cluster_demands(clusters)
        
        return [cluster for cluster in clusters if cluster]
    
    def _balance_cluster_demands(self, clusters: List[List[int]]):
        """平衡聚类间的需求"""
        max_iterations = 10
        vehicle_capacity = self.instance.vehicle_capacity
        
        for _ in range(max_iterations):
            # 计算每个聚类的总需求
            cluster_demands = []
            for cluster in clusters:
                total_demand = sum(self.instance.customers[c]['demand'] for c in cluster)
                cluster_demands.append(total_demand)
            
            # 找到需求最高和最低的聚类
            max_demand_idx = np.argmax(cluster_demands)
            min_demand_idx = np.argmin(cluster_demands)
            
            # 如果差异不大，停止调整
            if cluster_demands[max_demand_idx] - cluster_demands[min_demand_idx] < vehicle_capacity * 0.2:
                break
            
            # 从高需求聚类移动一个客户到低需求聚类
            if clusters[max_demand_idx]:
                # 选择需求最小的客户移动
                customer_to_move = min(clusters[max_demand_idx], 
                                     key=lambda c: self.instance.customers[c]['demand'])
                clusters[max_demand_idx].remove(customer_to_move)
                clusters[min_demand_idx].append(customer_to_move)
    
    def get_deep_instance_data(self) -> Dict[str, Any]:
        """获取Deep空间实例数据"""
        if not self.deep_nodes:
            raise ValueError("请先构造Deep空间")
        
        # 构造Deep空间的客户数据
        deep_customers = {}
        for i, node in enumerate(self.deep_nodes):
            pos_info = node.get_position_info()
            time_info = node.get_time_info()
            service_info = node.get_service_info()
            
            deep_customers[i] = {
                'x': pos_info['center'][0],
                'y': pos_info['center'][1],
                'start_x': pos_info['start_position'][0],
                'start_y': pos_info['start_position'][1],
                'end_x': pos_info['end_position'][0],
                'end_y': pos_info['end_position'][1],
                'demand': service_info['total_demand'],
                'ready_time': time_info['ready_time'],
                'due_time': time_info['due_time'],
                'service_time': time_info['internal_service_time'],
                'customer_count': service_info['customer_count'],
                'urgency': time_info['urgency'],
                'original_customers': service_info['customers']
            }
        
        # 构造Deep空间的车辆信息（继承原始实例）
        deep_vehicle = {
            'capacity': self.instance.vehicle_capacity,
            'x': self.instance.depot['x'],
            'y': self.instance.depot['y']
        }
        
        return {
            'customer': deep_customers,
            'vehicle': deep_vehicle,
            'depot': self.instance.depot.copy(),
            'node_mapping': self.node_mapping.copy(),
            'original_instance': self.instance
        }
    
    def visualize_deep_space(self, save_path: str = None):
        """可视化Deep空间"""
        if not self.deep_nodes:
            raise ValueError("请先构造Deep空间")
        
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 左图：原始客户分布
        ax1.set_title('原始客户分布')
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.deep_nodes)))
        
        for i, node in enumerate(self.deep_nodes):
            customers = node.customers
            x_coords = [self.instance.customers[c]['x'] for c in customers]
            y_coords = [self.instance.customers[c]['y'] for c in customers]
            ax1.scatter(x_coords, y_coords, c=[colors[i]], label=f'聚类{i}', alpha=0.7)
        
        # 绘制仓库
        ax1.scatter(self.instance.depot['x'], self.instance.depot['y'], 
                   c='red', marker='s', s=100, label='仓库')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右图：Deep空间节点
        ax2.set_title('Deep空间节点（增强位置信息）')
        
        for i, node in enumerate(self.deep_nodes):
            pos_info = node.get_position_info()
            
            # 绘制中心点
            ax2.scatter(pos_info['center'][0], pos_info['center'][1], 
                       c=[colors[i]], s=100, marker='o', label=f'节点{i}中心')
            
            # 绘制开始位置
            ax2.scatter(pos_info['start_position'][0], pos_info['start_position'][1], 
                       c=[colors[i]], s=60, marker='^', alpha=0.7)
            
            # 绘制结束位置
            ax2.scatter(pos_info['end_position'][0], pos_info['end_position'][1], 
                       c=[colors[i]], s=60, marker='v', alpha=0.7)
            
            # 连接开始和结束位置
            ax2.plot([pos_info['start_position'][0], pos_info['end_position'][0]], 
                    [pos_info['start_position'][1], pos_info['end_position'][1]], 
                    c=colors[i], alpha=0.5, linestyle='--')
        
        # 绘制仓库
        ax2.scatter(self.instance.depot['x'], self.instance.depot['y'], 
                   c='red', marker='s', s=100, label='仓库')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# 使用示例
if __name__ == "__main__":
    from vrptw_instance import VRPTWInstance
    
    # 加载实例
    instance = VRPTWInstance("data/RC1_2_1.txt")
    
    # 创建增强的Deep空间构造器
    constructor = EnhancedDeepConstructor(instance, cluster_method='time_spatial')
    
    # 构造Deep空间
    deep_nodes = constructor.construct_deep_space(target_clusters=8)
    
    print(f"构造了 {len(deep_nodes)} 个Deep节点")
    for node in deep_nodes:
        print(f"节点 {node.node_id}: {len(node.customers)} 个客户, 需求 {node.total_demand}")
        print(f"  位置信息: {node.get_position_info()}")
        print(f"  时间信息: {node.get_time_info()}")
        print()
    
    # 可视化
    constructor.visualize_deep_space()