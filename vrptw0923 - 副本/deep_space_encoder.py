#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep空间状态编码器
将Deep空间的状态信息编码为神经网络可处理的特征向量
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


class DeepSpaceStateEncoder:
    """Deep空间状态编码器"""
    
    def __init__(self, enhanced_constructor, feature_dim: int = 128, 
                 use_normalization: bool = True, use_pca: bool = False):
        """
        初始化编码器
        
        Args:
            enhanced_constructor: 增强的Deep空间构造器
            feature_dim: 特征维度
            use_normalization: 是否使用归一化
            use_pca: 是否使用PCA降维
        """
        self.constructor = enhanced_constructor
        self.deep_nodes = enhanced_constructor.deep_nodes
        self.instance = enhanced_constructor.instance
        self.feature_dim = feature_dim
        self.use_normalization = use_normalization
        self.use_pca = use_pca
        
        # 编码器组件
        self.node_encoder = DeepNodeEncoder()
        self.route_encoder = DeepRouteEncoder()
        self.global_encoder = DeepGlobalEncoder()
        
        # 数据预处理器
        self.scaler = StandardScaler() if use_normalization else None
        self.pca = PCA(n_components=min(feature_dim, 50)) if use_pca else None
        
        # 特征缓存
        self.feature_cache = {}
        self.is_fitted = False
    
    def encode_state(self, individual_chromosome: List[List[int]], 
                    current_solution: Optional[List[List[int]]] = None) -> np.ndarray:
        """
        编码Deep空间状态
        
        Args:
            individual_chromosome: Deep空间个体染色体
            current_solution: 当前原始空间解（可选）
            
        Returns:
            编码后的状态向量
        """
        # 节点级特征
        node_features = self._encode_node_features(individual_chromosome)
        
        # 路径级特征
        route_features = self._encode_route_features(individual_chromosome)
        
        # 全局特征
        global_features = self._encode_global_features(individual_chromosome, current_solution)
        
        # 关系特征
        relation_features = self._encode_relation_features(individual_chromosome)
        
        # 合并所有特征
        combined_features = np.concatenate([
            node_features.flatten(),
            route_features.flatten(),
            global_features.flatten(),
            relation_features.flatten()
        ])
        
        # 应用预处理
        if self.is_fitted:
            processed_features = self._apply_preprocessing(combined_features)
        else:
            processed_features = combined_features
        
        # 调整到目标维度
        final_features = self._adjust_feature_dimension(processed_features)
        
        return final_features
    
    def _encode_node_features(self, chromosome: List[List[int]]) -> np.ndarray:
        """编码节点级特征"""
        node_count = len(self.deep_nodes)
        node_features = np.zeros((node_count, 15))  # 每个节点15个特征
        
        # 节点使用状态
        used_nodes = set()
        for route in chromosome:
            used_nodes.update(route)
        
        for i, deep_node in enumerate(self.deep_nodes):
            # 基本特征
            pos_info = deep_node.get_position_info()
            time_info = deep_node.get_time_info()
            service_info = deep_node.get_service_info()
            
            features = [
                # 位置特征 (6个)
                pos_info['center'][0] / 1000,  # 归一化坐标
                pos_info['center'][1] / 1000,
                pos_info['start_position'][0] / 1000,
                pos_info['start_position'][1] / 1000,
                pos_info['end_position'][0] / 1000,
                pos_info['end_position'][1] / 1000,
                
                # 时间特征 (4个)
                time_info['ready_time'] / 1000,  # 归一化时间
                time_info['due_time'] / 1000,
                time_info['internal_service_time'] / 100,
                time_info['urgency'],
                
                # 服务特征 (3个)
                service_info['total_demand'] / 100,  # 归一化需求
                service_info['customer_count'] / 10,
                pos_info['radius'] / 100,
                
                # 状态特征 (2个)
                1.0 if i in used_nodes else 0.0,  # 是否被使用
                self._calculate_node_accessibility(deep_node)  # 可达性
            ]
            
            node_features[i] = features
        
        return node_features
    
    def _encode_route_features(self, chromosome: List[List[int]]) -> np.ndarray:
        """编码路径级特征"""
        max_routes = 10  # 最大路径数
        route_features = np.zeros((max_routes, 12))  # 每条路径12个特征
        
        for i, route in enumerate(chromosome[:max_routes]):
            if not route:
                continue
            
            # 路径基本特征
            route_length = len(route)
            total_demand = sum(self.deep_nodes[node_id].total_demand 
                             for node_id in route if node_id < len(self.deep_nodes))
            
            # 路径时间特征
            route_ready_time = min(self.deep_nodes[node_id].ready_time 
                                 for node_id in route if node_id < len(self.deep_nodes))
            route_due_time = max(self.deep_nodes[node_id].due_time 
                               for node_id in route if node_id < len(self.deep_nodes))
            
            # 路径距离特征
            route_distance = self._calculate_route_distance(route)
            
            # 路径紧凑性
            compactness = self._calculate_route_compactness(route)
            
            # 路径平衡性
            balance = self._calculate_route_balance(route)
            
            features = [
                route_length / 10,  # 归一化路径长度
                total_demand / 200,  # 归一化总需求
                route_ready_time / 1000,  # 归一化时间
                route_due_time / 1000,
                (route_due_time - route_ready_time) / 1000,  # 时间窗跨度
                route_distance / 1000,  # 归一化距离
                compactness,
                balance,
                self._calculate_route_urgency(route),  # 路径紧急度
                self._calculate_route_feasibility(route),  # 路径可行性
                self._calculate_route_efficiency(route),  # 路径效率
                1.0  # 路径存在标志
            ]
            
            route_features[i] = features
        
        return route_features
    
    def _encode_global_features(self, chromosome: List[List[int]], 
                              current_solution: Optional[List[List[int]]]) -> np.ndarray:
        """编码全局特征"""
        global_features = np.zeros(20)  # 20个全局特征
        
        # 基本统计特征
        total_routes = len([route for route in chromosome if route])
        total_nodes_used = len(set(node for route in chromosome for node in route))
        total_nodes_available = len(self.deep_nodes)
        
        # 负载特征
        total_demand = sum(self.deep_nodes[node_id].total_demand 
                         for route in chromosome for node_id in route 
                         if node_id < len(self.deep_nodes))
        
        # 时间特征
        earliest_time = min((self.deep_nodes[node_id].ready_time 
                           for route in chromosome for node_id in route 
                           if node_id < len(self.deep_nodes)), default=0)
        latest_time = max((self.deep_nodes[node_id].due_time 
                         for route in chromosome for node_id in route 
                         if node_id < len(self.deep_nodes)), default=1000)
        
        # 空间分布特征
        spatial_spread = self._calculate_spatial_spread(chromosome)
        
        # 解的质量特征（如果有当前解）
        solution_quality = 0.0
        if current_solution:
            solution_quality = self._estimate_solution_quality(current_solution)
        
        features = [
            total_routes / 10,  # 归一化路径数
            total_nodes_used / total_nodes_available,  # 节点使用率
            total_demand / (self.instance.vehicle_capacity * 10),  # 需求密度
            earliest_time / 1000,  # 归一化时间
            latest_time / 1000,
            (latest_time - earliest_time) / 1000,  # 时间跨度
            spatial_spread,  # 空间分布
            self._calculate_load_balance(chromosome),  # 负载平衡
            self._calculate_time_balance(chromosome),  # 时间平衡
            self._calculate_compactness_score(chromosome),  # 紧凑性得分
            solution_quality,  # 解质量
            self._calculate_diversity_score(chromosome),  # 多样性得分
            self._calculate_feasibility_score(chromosome),  # 可行性得分
            self._calculate_efficiency_score(chromosome),  # 效率得分
            self._calculate_stability_score(chromosome),  # 稳定性得分
            # 预留特征
            0.0, 0.0, 0.0, 0.0, 0.0
        ]
        
        return np.array(features)
    
    def _encode_relation_features(self, chromosome: List[List[int]]) -> np.ndarray:
        """编码关系特征"""
        node_count = len(self.deep_nodes)
        relation_features = np.zeros((node_count, node_count))
        
        # 路径内关系
        for route in chromosome:
            for i in range(len(route)):
                for j in range(len(route)):
                    if i != j and route[i] < node_count and route[j] < node_count:
                        # 同路径关系强度
                        relation_features[route[i]][route[j]] = 1.0 / (abs(i - j) + 1)
        
        # 路径间关系
        for i, route1 in enumerate(chromosome):
            for j, route2 in enumerate(chromosome):
                if i != j:
                    for node1 in route1:
                        for node2 in route2:
                            if node1 < node_count and node2 < node_count:
                                # 不同路径间的关系（基于距离）
                                distance = self._calculate_node_distance(node1, node2)
                                relation_features[node1][node2] = max(
                                    relation_features[node1][node2],
                                    1.0 / (1.0 + distance / 100)
                                )
        
        return relation_features
    
    def _calculate_node_accessibility(self, deep_node) -> float:
        """计算节点可达性"""
        # 基于时间窗和位置的可达性评估
        time_window_size = deep_node.due_time - deep_node.ready_time
        urgency = deep_node.urgency
        
        # 可达性 = 时间窗大小的倒数 + 紧急度
        accessibility = 1.0 / (1.0 + time_window_size / 100) + urgency
        return min(accessibility, 1.0)
    
    def _calculate_route_distance(self, route: List[int]) -> float:
        """计算路径距离"""
        if len(route) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(route) - 1):
            if route[i] < len(self.deep_nodes) and route[i+1] < len(self.deep_nodes):
                distance = self.deep_nodes[route[i]].distance_to_node(
                    self.deep_nodes[route[i+1]], mode='end_to_start')
                total_distance += distance
        
        return total_distance
    
    def _calculate_route_compactness(self, route: List[int]) -> float:
        """计算路径紧凑性"""
        if len(route) < 2:
            return 1.0
        
        # 计算路径的空间紧凑性
        positions = []
        for node_id in route:
            if node_id < len(self.deep_nodes):
                pos_info = self.deep_nodes[node_id].get_position_info()
                positions.append(pos_info['center'])
        
        if len(positions) < 2:
            return 1.0
        
        # 计算凸包面积与总距离的比值
        total_distance = sum(math.sqrt((positions[i][0] - positions[i+1][0])**2 + 
                                     (positions[i][1] - positions[i+1][1])**2)
                           for i in range(len(positions) - 1))
        
        # 计算边界框面积
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        bbox_area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
        
        if bbox_area == 0:
            return 1.0
        
        compactness = total_distance / (bbox_area + 1)
        return min(compactness, 1.0)
    
    def _calculate_route_balance(self, route: List[int]) -> float:
        """计算路径平衡性"""
        if not route:
            return 1.0
        
        # 计算需求平衡性
        demands = [self.deep_nodes[node_id].total_demand 
                  for node_id in route if node_id < len(self.deep_nodes)]
        
        if not demands:
            return 1.0
        
        mean_demand = np.mean(demands)
        if mean_demand == 0:
            return 1.0
        
        variance = np.var(demands)
        balance = 1.0 / (1.0 + variance / (mean_demand**2))
        
        return balance
    
    def _calculate_route_urgency(self, route: List[int]) -> float:
        """计算路径紧急度"""
        if not route:
            return 0.0
        
        urgencies = [self.deep_nodes[node_id].urgency 
                    for node_id in route if node_id < len(self.deep_nodes)]
        
        return np.mean(urgencies) if urgencies else 0.0
    
    def _calculate_route_feasibility(self, route: List[int]) -> float:
        """计算路径可行性"""
        if not route:
            return 1.0
        
        # 简化的可行性检查
        total_demand = sum(self.deep_nodes[node_id].total_demand 
                         for node_id in route if node_id < len(self.deep_nodes))
        
        capacity_feasibility = min(1.0, self.instance.vehicle_capacity / (total_demand + 1))
        
        # 时间窗可行性（简化）
        time_feasibility = 1.0  # 这里可以添加更复杂的时间窗检查
        
        return min(capacity_feasibility, time_feasibility)
    
    def _calculate_route_efficiency(self, route: List[int]) -> float:
        """计算路径效率"""
        if not route:
            return 0.0
        
        route_distance = self._calculate_route_distance(route)
        total_demand = sum(self.deep_nodes[node_id].total_demand 
                         for node_id in route if node_id < len(self.deep_nodes))
        
        if route_distance == 0:
            return 1.0
        
        efficiency = total_demand / (route_distance + 1)
        return min(efficiency / 10, 1.0)  # 归一化
    
    def _calculate_spatial_spread(self, chromosome: List[List[int]]) -> float:
        """计算空间分布"""
        all_positions = []
        for route in chromosome:
            for node_id in route:
                if node_id < len(self.deep_nodes):
                    pos_info = self.deep_nodes[node_id].get_position_info()
                    all_positions.append(pos_info['center'])
        
        if len(all_positions) < 2:
            return 0.0
        
        # 计算位置的标准差
        x_coords = [pos[0] for pos in all_positions]
        y_coords = [pos[1] for pos in all_positions]
        
        x_std = np.std(x_coords)
        y_std = np.std(y_coords)
        
        spread = math.sqrt(x_std**2 + y_std**2) / 1000  # 归一化
        return min(spread, 1.0)
    
    def _calculate_load_balance(self, chromosome: List[List[int]]) -> float:
        """计算负载平衡"""
        route_demands = []
        for route in chromosome:
            if route:
                demand = sum(self.deep_nodes[node_id].total_demand 
                           for node_id in route if node_id < len(self.deep_nodes))
                route_demands.append(demand)
        
        if not route_demands:
            return 1.0
        
        mean_demand = np.mean(route_demands)
        if mean_demand == 0:
            return 1.0
        
        variance = np.var(route_demands)
        balance = 1.0 / (1.0 + variance / (mean_demand**2))
        
        return balance
    
    def _calculate_time_balance(self, chromosome: List[List[int]]) -> float:
        """计算时间平衡"""
        route_times = []
        for route in chromosome:
            if route:
                total_time = sum(self.deep_nodes[node_id].internal_service_time 
                               for node_id in route if node_id < len(self.deep_nodes))
                route_times.append(total_time)
        
        if not route_times:
            return 1.0
        
        mean_time = np.mean(route_times)
        if mean_time == 0:
            return 1.0
        
        variance = np.var(route_times)
        balance = 1.0 / (1.0 + variance / (mean_time**2))
        
        return balance
    
    def _calculate_compactness_score(self, chromosome: List[List[int]]) -> float:
        """计算紧凑性得分"""
        compactness_scores = []
        for route in chromosome:
            if route:
                compactness = self._calculate_route_compactness(route)
                compactness_scores.append(compactness)
        
        return np.mean(compactness_scores) if compactness_scores else 0.0
    
    def _estimate_solution_quality(self, solution: List[List[int]]) -> float:
        """估计解质量"""
        # 这里可以使用解码器来评估解质量
        # 简化版本：基于路径数量和总距离的估计
        total_routes = len([route for route in solution if route])
        total_customers = sum(len(route) for route in solution)
        
        if total_customers == 0:
            return 0.0
        
        efficiency = total_customers / (total_routes + 1)
        return min(efficiency / 10, 1.0)
    
    def _calculate_diversity_score(self, chromosome: List[List[int]]) -> float:
        """计算多样性得分"""
        # 基于路径长度的多样性
        route_lengths = [len(route) for route in chromosome if route]
        
        if not route_lengths:
            return 0.0
        
        if len(route_lengths) == 1:
            return 0.0
        
        variance = np.var(route_lengths)
        mean_length = np.mean(route_lengths)
        
        if mean_length == 0:
            return 0.0
        
        diversity = variance / (mean_length**2)
        return min(diversity, 1.0)
    
    def _calculate_feasibility_score(self, chromosome: List[List[int]]) -> float:
        """计算可行性得分"""
        feasibility_scores = []
        for route in chromosome:
            if route:
                feasibility = self._calculate_route_feasibility(route)
                feasibility_scores.append(feasibility)
        
        return np.mean(feasibility_scores) if feasibility_scores else 0.0
    
    def _calculate_efficiency_score(self, chromosome: List[List[int]]) -> float:
        """计算效率得分"""
        efficiency_scores = []
        for route in chromosome:
            if route:
                efficiency = self._calculate_route_efficiency(route)
                efficiency_scores.append(efficiency)
        
        return np.mean(efficiency_scores) if efficiency_scores else 0.0
    
    def _calculate_stability_score(self, chromosome: List[List[int]]) -> float:
        """计算稳定性得分"""
        # 基于路径结构的稳定性评估
        # 这里使用路径长度的一致性作为稳定性指标
        route_lengths = [len(route) for route in chromosome if route]
        
        if not route_lengths:
            return 1.0
        
        if len(route_lengths) == 1:
            return 1.0
        
        std_dev = np.std(route_lengths)
        mean_length = np.mean(route_lengths)
        
        if mean_length == 0:
            return 1.0
        
        stability = 1.0 / (1.0 + std_dev / mean_length)
        return stability
    
    def _calculate_node_distance(self, node1_id: int, node2_id: int) -> float:
        """计算两个节点间的距离"""
        if node1_id >= len(self.deep_nodes) or node2_id >= len(self.deep_nodes):
            return float('inf')
        
        return self.deep_nodes[node1_id].distance_to_node(
            self.deep_nodes[node2_id], mode='center_to_center')
    
    def _apply_preprocessing(self, features: np.ndarray) -> np.ndarray:
        """应用预处理"""
        processed = features.copy()
        
        # 标准化
        if self.scaler is not None:
            processed = self.scaler.transform(processed.reshape(1, -1)).flatten()
        
        # PCA降维
        if self.pca is not None:
            processed = self.pca.transform(processed.reshape(1, -1)).flatten()
        
        return processed
    
    def _adjust_feature_dimension(self, features: np.ndarray) -> np.ndarray:
        """调整特征维度"""
        if len(features) == self.feature_dim:
            return features
        elif len(features) > self.feature_dim:
            # 截断
            return features[:self.feature_dim]
        else:
            # 填充
            padded = np.zeros(self.feature_dim)
            padded[:len(features)] = features
            return padded
    
    def fit_preprocessing(self, sample_chromosomes: List[List[List[int]]]):
        """拟合预处理器"""
        if not sample_chromosomes:
            return
        
        # 收集样本特征
        sample_features = []
        for chromosome in sample_chromosomes:
            try:
                features = self.encode_state(chromosome)
                sample_features.append(features)
            except Exception as e:
                continue
        
        if not sample_features:
            return
        
        sample_features = np.array(sample_features)
        
        # 拟合标准化器
        if self.scaler is not None:
            self.scaler.fit(sample_features)
        
        # 拟合PCA
        if self.pca is not None:
            if self.scaler is not None:
                normalized_features = self.scaler.transform(sample_features)
            else:
                normalized_features = sample_features
            self.pca.fit(normalized_features)
        
        self.is_fitted = True
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        # 这里可以实现特征重要性分析
        # 简化版本：返回各类特征的权重
        return {
            'node_features': 0.4,
            'route_features': 0.3,
            'global_features': 0.2,
            'relation_features': 0.1
        }


class DeepNodeEncoder:
    """Deep节点编码器"""
    
    def encode_node(self, deep_node, context: Dict[str, Any] = None) -> np.ndarray:
        """编码单个Deep节点"""
        # 实现节点编码逻辑
        pass


class DeepRouteEncoder:
    """Deep路径编码器"""
    
    def encode_route(self, route: List[int], deep_nodes: List, 
                    context: Dict[str, Any] = None) -> np.ndarray:
        """编码单条路径"""
        # 实现路径编码逻辑
        pass


class DeepGlobalEncoder:
    """Deep全局编码器"""
    
    def encode_global_state(self, chromosome: List[List[int]], 
                          context: Dict[str, Any] = None) -> np.ndarray:
        """编码全局状态"""
        # 实现全局状态编码逻辑
        pass


# 使用示例
if __name__ == "__main__":
    from vrptw_instance import VRPTWInstance
    from enhanced_deep_constructor import EnhancedDeepConstructor
    
    # 加载实例
    instance = VRPTWInstance("data/RC1_2_1.txt")
    
    # 创建Deep空间构造器
    constructor = EnhancedDeepConstructor(instance, cluster_method='time_spatial')
    deep_nodes = constructor.construct_deep_space(target_clusters=8)
    
    # 创建状态编码器
    encoder = DeepSpaceStateEncoder(constructor, feature_dim=128)
    
    # 示例染色体
    sample_chromosome = [[0, 1, 2], [3, 4], [5, 6, 7]]
    
    # 编码状态
    state_vector = encoder.encode_state(sample_chromosome)
    
    print(f"状态向量维度: {state_vector.shape}")
    print(f"状态向量: {state_vector[:10]}...")  # 显示前10个特征
    
    # 批量拟合预处理器
    sample_chromosomes = [
        [[0, 1], [2, 3], [4, 5]],
        [[0, 2, 4], [1, 3, 5]],
        [[0], [1], [2], [3], [4], [5]]
    ]
    
    encoder.fit_preprocessing(sample_chromosomes)
    
    # 重新编码
    processed_state_vector = encoder.encode_state(sample_chromosome)
    print(f"预处理后状态向量: {processed_state_vector[:10]}...")
    
    # 特征重要性
    importance = encoder.get_feature_importance()
    print(f"特征重要性: {importance}")