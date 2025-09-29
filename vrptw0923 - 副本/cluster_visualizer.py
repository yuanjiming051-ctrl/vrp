"""
节点聚合可视化模块
用于可视化Deep空间重构后的节点聚合情况，展示节点间的连接关系和层级结构
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import networkx as nx
from matplotlib.colors import ListedColormap
import os
from datetime import datetime


class ClusterVisualizer:
    """节点聚合可视化器"""
    
    def __init__(self, output_dir="visualization_output"):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        self.ensure_output_dir()
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 颜色配置
        self.colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
            '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2'
        ]
        
    def ensure_output_dir(self):
        """确保输出目录存在"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def visualize_cluster_aggregation(self, instance, deep_inst, cluster_labels, 
                                    round_idx, best_full_solution=None, 
                                    title_suffix=""):
        """
        可视化节点聚合情况
        
        Args:
            instance: 原始VRPTW实例
            deep_inst: Deep空间实例
            cluster_labels: 聚类标签数组
            round_idx: 当前轮次
            best_full_solution: 最优Full解（可选）
            title_suffix: 标题后缀
        """
        try:
            # 创建图形
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            # 左图：原始节点分布和聚类结果
            self._plot_original_clustering(ax1, instance, cluster_labels, best_full_solution)
            
            # 右图：Deep空间聚合后的网络结构
            self._plot_deep_network(ax2, instance, deep_inst, cluster_labels)
            
            # 设置总标题
            timestamp = datetime.now().strftime("%H:%M:%S")
            main_title = f"第{round_idx}轮 Deep空间节点聚合可视化 ({timestamp})"
            if title_suffix:
                main_title += f" - {title_suffix}"
            fig.suptitle(main_title, fontsize=16, fontweight='bold')
            
            # 调整布局
            plt.tight_layout()
            plt.subplots_adjust(top=0.93)
            
            # 保存图片
            filename = f"cluster_round_{round_idx}_{timestamp.replace(':', '-')}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            
            print(f"  [可视化] 节点聚合图已保存: {filepath}")
            
            # 显示图片（可选）
            plt.show()
            plt.close()
            
            return filepath
            
        except Exception as e:
            print(f"  [可视化错误] 生成聚合可视化失败: {str(e)}")
            return None
    
    def _plot_original_clustering(self, ax, instance, cluster_labels, best_full_solution):
        """绘制原始节点分布和聚类结果"""
        # 获取节点坐标
        coords = np.array([[c.x, c.y] for c in instance.customers])
        depot_coord = [instance.depot.x, instance.depot.y]
        
        # 获取唯一的聚类标签
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels)
        
        # 绘制聚类
        for i, label in enumerate(unique_labels):
            if label == -1:  # 噪声点
                mask = cluster_labels == label
                ax.scatter(coords[mask, 0], coords[mask, 1], 
                          c='black', marker='x', s=50, alpha=0.6, label='噪声点')
            else:
                mask = cluster_labels == label
                color = self.colors[i % len(self.colors)]
                ax.scatter(coords[mask, 0], coords[mask, 1], 
                          c=color, s=60, alpha=0.7, label=f'聚类 {label}')
                
                # 绘制聚类边界（凸包）
                if np.sum(mask) > 2:
                    cluster_coords = coords[mask]
                    try:
                        from scipy.spatial import ConvexHull
                        hull = ConvexHull(cluster_coords)
                        for simplex in hull.simplices:
                            ax.plot(cluster_coords[simplex, 0], cluster_coords[simplex, 1], 
                                   color=color, alpha=0.3, linewidth=1)
                    except:
                        pass
        
        # 绘制仓库
        ax.scatter(depot_coord[0], depot_coord[1], c='red', marker='s', 
                  s=200, label='仓库', edgecolors='black', linewidth=2)
        
        # 如果有最优解，绘制路径
        if best_full_solution is not None:
            self._draw_solution_routes(ax, instance, best_full_solution, alpha=0.3)
        
        # 添加节点编号
        for i, (x, y) in enumerate(coords):
            ax.annotate(str(i+1), (x, y), xytext=(3, 3), textcoords='offset points',
                       fontsize=8, alpha=0.7)
        
        ax.set_title(f'原始节点分布与聚类结果\n(共{n_clusters}个聚类)', fontsize=12, fontweight='bold')
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_deep_network(self, ax, instance, deep_inst, cluster_labels):
        """绘制Deep空间聚合后的网络结构"""
        # 创建网络图
        G = nx.Graph()
        
        # 获取原始坐标和Deep坐标
        original_coords = np.array([[c.x, c.y] for c in instance.customers])
        deep_coords = np.array([[c.x, c.y] for c in deep_inst.customers])
        depot_coord = [instance.depot.x, instance.depot.y]
        
        # 添加仓库节点
        G.add_node('depot', pos=depot_coord, node_type='depot')
        
        # 添加Deep节点
        for i, (x, y) in enumerate(deep_coords):
            G.add_node(f'deep_{i}', pos=[x, y], node_type='deep')
        
        # 计算每个Deep节点包含的原始节点
        unique_labels = np.unique(cluster_labels)
        deep_node_mapping = {}
        
        for deep_idx, label in enumerate(unique_labels):
            if label != -1:  # 忽略噪声点
                mask = cluster_labels == label
                original_indices = np.where(mask)[0]
                deep_node_mapping[deep_idx] = original_indices
        
        # 绘制Deep节点
        deep_pos = {f'deep_{i}': [x, y] for i, (x, y) in enumerate(deep_coords)}
        deep_pos['depot'] = depot_coord
        
        # 绘制仓库
        ax.scatter(depot_coord[0], depot_coord[1], c='red', marker='s', 
                  s=300, label='仓库', edgecolors='black', linewidth=3, zorder=5)
        
        # 绘制Deep节点和连接
        for deep_idx, (x, y) in enumerate(deep_coords):
            # 计算节点大小（基于包含的原始节点数量）
            if deep_idx in deep_node_mapping:
                node_count = len(deep_node_mapping[deep_idx])
                node_size = 100 + node_count * 20
                
                # 绘制Deep节点
                color = self.colors[deep_idx % len(self.colors)]
                ax.scatter(x, y, c=color, s=node_size, alpha=0.8, 
                          edgecolors='black', linewidth=2, zorder=4)
                
                # 添加节点标签（显示包含的原始节点数量）
                ax.annotate(f'D{deep_idx}\n({node_count}个)', (x, y), 
                           ha='center', va='center', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                # 绘制到原始节点的连接线
                for orig_idx in deep_node_mapping[deep_idx]:
                    orig_x, orig_y = original_coords[orig_idx]
                    ax.plot([x, orig_x], [y, orig_y], color=color, alpha=0.4, 
                           linewidth=1, linestyle='--', zorder=1)
                    
                    # 绘制原始节点（小点）
                    ax.scatter(orig_x, orig_y, c=color, s=30, alpha=0.6, 
                              edgecolors='black', linewidth=0.5, zorder=2)
                    
                    # 添加原始节点编号
                    ax.annotate(str(orig_idx+1), (orig_x, orig_y), 
                               xytext=(2, 2), textcoords='offset points',
                               fontsize=7, alpha=0.8)
        
        # 绘制Deep节点之间的潜在连接（基于距离）
        for i in range(len(deep_coords)):
            for j in range(i+1, len(deep_coords)):
                dist = np.linalg.norm(deep_coords[i] - deep_coords[j])
                # 如果距离较近，绘制连接线
                if dist < np.mean([np.linalg.norm(deep_coords[k] - deep_coords[l]) 
                                  for k in range(len(deep_coords)) 
                                  for l in range(k+1, len(deep_coords))]) * 1.2:
                    ax.plot([deep_coords[i][0], deep_coords[j][0]], 
                           [deep_coords[i][1], deep_coords[j][1]], 
                           color='gray', alpha=0.3, linewidth=1, zorder=0)
        
        ax.set_title(f'Deep空间聚合网络结构\n(原始{len(original_coords)}个节点 → Deep{len(deep_coords)}个节点)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                      markersize=15, label='仓库'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                      markersize=10, label='Deep节点'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                      markersize=6, label='原始节点'),
            plt.Line2D([0], [0], color='gray', linestyle='--', alpha=0.6, label='聚合连接'),
            plt.Line2D([0], [0], color='gray', alpha=0.3, label='潜在路径')
        ]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    def _draw_solution_routes(self, ax, instance, solution, alpha=0.5):
        """绘制解的路径"""
        depot_coord = [instance.depot.x, instance.depot.y]
        
        for route_idx, route in enumerate(solution):
            if len(route) == 0:
                continue
                
            color = self.colors[route_idx % len(self.colors)]
            
            # 绘制从仓库到第一个客户的路径
            first_customer = instance.customers[route[0]]
            ax.plot([depot_coord[0], first_customer.x], 
                   [depot_coord[1], first_customer.y], 
                   color=color, alpha=alpha, linewidth=2)
            
            # 绘制客户之间的路径
            for i in range(len(route) - 1):
                curr_customer = instance.customers[route[i]]
                next_customer = instance.customers[route[i + 1]]
                ax.plot([curr_customer.x, next_customer.x], 
                       [curr_customer.y, next_customer.y], 
                       color=color, alpha=alpha, linewidth=2)
            
            # 绘制从最后一个客户回到仓库的路径
            last_customer = instance.customers[route[-1]]
            ax.plot([last_customer.x, depot_coord[0]], 
                   [last_customer.y, depot_coord[1]], 
                   color=color, alpha=alpha, linewidth=2)
    
    def create_summary_report(self, round_idx, original_nodes, deep_nodes, 
                            cluster_info, performance_metrics=None):
        """
        创建聚合摘要报告
        
        Args:
            round_idx: 轮次
            original_nodes: 原始节点数
            deep_nodes: Deep节点数
            cluster_info: 聚类信息字典
            performance_metrics: 性能指标（可选）
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            report = f"""
=== Deep空间节点聚合报告 ===
时间: {timestamp}
轮次: {round_idx}

聚合统计:
- 原始节点数: {original_nodes}
- Deep节点数: {deep_nodes}
- 压缩比: {original_nodes/deep_nodes:.2f}:1
- 聚类数量: {cluster_info.get('n_clusters', 'N/A')}
- 平均聚类大小: {cluster_info.get('avg_cluster_size', 'N/A'):.2f}
- 最大聚类大小: {cluster_info.get('max_cluster_size', 'N/A')}
- 最小聚类大小: {cluster_info.get('min_cluster_size', 'N/A')}
"""
            
            if performance_metrics:
                report += f"""
性能指标:
- 聚合前成本: {performance_metrics.get('before_cost', 'N/A'):.2f}
- 聚合后成本: {performance_metrics.get('after_cost', 'N/A'):.2f}
- 成本变化: {performance_metrics.get('cost_change', 'N/A'):.2f}
"""
            
            # 保存报告
            report_filename = f"cluster_report_round_{round_idx}.txt"
            report_filepath = os.path.join(self.output_dir, report_filename)
            
            with open(report_filepath, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"  [报告] 聚合报告已保存: {report_filepath}")
            
            return report_filepath
            
        except Exception as e:
            print(f"  [报告错误] 生成聚合报告失败: {str(e)}")
            return None