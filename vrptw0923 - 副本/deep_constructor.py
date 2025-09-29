# File: deep_constructor.py

import numpy as np
from sklearn.manifold import MDS
from route_optimizer import two_opt_route


class DeepConstructor:
    """
    DeepConstructor：将原空间客户集递归聚类（基于距离），
    得到一批"深度子路线"（每个子路线是一组原客户索引）。
    最后将这些子路线聚合成"深度节点"，输出 deepdata。
    """

    def __init__(self, instance, delta=None):
        """
        :param instance: VRPTWInstance，包含 ordinary_customers、distance_matrix、vehicle_info
        :param delta: 保留参数（已不使用）
        """
        self.instance = instance
        self.cust = instance.ordinary_customers
        self.D = instance.distance_matrix
        self.cap = instance.vehicle_info['capacity']
        self.n = len(self.cust)
        self.eps = np.finfo(float).eps
        self.VC_new = []  # 存储所有可行子路线（子簇里的客户索引列表）

    def time_window_detect(self, route):
        """检查给定子路线（客户索引列表）是否满足时间窗约束"""
        if not route:
            return True
            
        # 从仓库(索引0)出发到第一个客户
        t = 0.0
        first_customer = self.cust[route[0]]
        t += self.D[0, route[0] + 1]  # 从仓库到第一个客户的时间
        
        # 检查第一个客户的时间窗约束
        if t > first_customer['due_date']:
            return False
        if t < first_customer['ready_time']:
            t = first_customer['ready_time']
        t += first_customer['service_time']
        
        # 检查路径中其余客户的时间窗约束
        for i in range(len(route) - 1):
            a = route[i] + 1  # 当前客户在距离矩阵中的索引
            b = route[i + 1] + 1  # 下一个客户在距离矩阵中的索引
            t += self.D[a, b]  # 旅行时间
            nxt = self.cust[route[i + 1]]
            
            # 检查到达时间是否超过截止时间
            if t > nxt['due_date']:
                return False
            # 如果早到，需要等待
            if t < nxt['ready_time']:
                t = nxt['ready_time']
            # 加上服务时间
            t += nxt['service_time']
        
        # 检查从最后一个客户返回仓库的时间约束
        # 假设仓库的截止时间是所有客户中最晚的截止时间
        last_customer_idx = route[-1] + 1
        return_time = t + self.D[last_customer_idx, 0]
        
        # 获取仓库的截止时间（通常在vehicle_info中或者是最大的due_date）
        depot_due_time = max(customer['due_date'] for customer in self.cust)
        
        return return_time <= depot_due_time

    def _distance_recursive_split(self, cluster):
        """
        基于客户间的距离，递归二分聚类
        """
        route = sorted(cluster, key=lambda i: self.cust[i]['due_date'])
        total_d = sum(self.cust[i]['demand'] for i in route)

        # 满足容量和时间窗约束 → 不再拆分
        if total_d <= self.cap and self.time_window_detect(route):
            return [route]

        if len(route) <= 1:
            return [route]

        # 找簇内最远的两个点作为种子
        coords = np.array([[self.cust[i]['xcoord'], self.cust[i]['ycoord']] for i in route])
        dist_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
        i1, i2 = np.unravel_index(np.argmax(dist_mat), dist_mat.shape)
        seed1, seed2 = route[i1], route[i2]

        # 按距离最近的种子划分
        sub0, sub1 = [], []
        for idx in route:
            d1 = np.hypot(self.cust[idx]['xcoord'] - self.cust[seed1]['xcoord'],
                          self.cust[idx]['ycoord'] - self.cust[seed1]['ycoord'])
            d2 = np.hypot(self.cust[idx]['xcoord'] - self.cust[seed2]['xcoord'],
                          self.cust[idx]['ycoord'] - self.cust[seed2]['ycoord'])
            if d1 <= d2:
                sub0.append(idx)
            else:
                sub1.append(idx)

        if not sub0 or not sub1:  # 无法分裂 → 返回原簇
            return [route]

        out = []
        out.extend(self._distance_recursive_split(sub0))
        out.extend(self._distance_recursive_split(sub1))
        return out

    def run(self):
        """
        执行从 0..n-1 客户 的深度构造：
        1. 计算JS散度矩阵（保留）
        2. MDS降一维生成时间特征（保留，暂未使用）
        3. 基于距离的递归聚类
        4. 子簇进行two_opt优化
        5. 聚合为deepdata
        """
        # === 计算 JS 散度矩阵 (仍然保留，但后续未直接用作聚类) ===
        JS = np.zeros((self.n, self.n))
        for i in range(self.n):
            p = np.array([self.cust[i]['ready_time'], self.cust[i]['due_date']], dtype=float)
            if p.sum() > 0:
                p /= p.sum()
            for j in range(i + 1, self.n):
                q = np.array([self.cust[j]['ready_time'], self.cust[j]['due_date']], dtype=float)
                if q.sum() > 0:
                    q /= q.sum()
                m = 0.5 * (p + q)
                kl1 = np.sum(p * np.log((p / (m + self.eps)) + self.eps))
                kl2 = np.sum(q * np.log((q / (m + self.eps)) + self.eps))
                JS[i, j] = JS[j, i] = 0.5 * (kl1 + kl2)

        # === MDS降到1维 (保留，但仅作为特征演示，不参与聚类) ===
        if self.n > 1:
            tf = MDS(n_components=1, dissimilarity='precomputed',
                     normalized_stress='auto').fit_transform(JS).flatten()
        else:
            tf = np.zeros(self.n)

        # === 距离递归聚类 ===
        self.VC_new.clear()
        clusters = self._distance_recursive_split(list(range(self.n)))
        for cluster in clusters:
            self.VC_new.append(cluster)

        # === 2-opt 改进 ===
        for idx, route in enumerate(self.VC_new):
            if 2 < len(route) <= 15:
                improved = two_opt_route(route, self.D)
                total_d = sum(self.cust[i]['demand'] for i in improved)
                if total_d <= self.cap and self.time_window_detect(improved):
                    self.VC_new[idx] = improved

        # === 聚合子路线为深度节点 ===
        deepdata = {
            'customer': [],
            'depot': self.instance.warehouse,
            'vehicle': self.instance.vehicle_info,
            'VC_new': self.VC_new
        }

        for route in self.VC_new:
            if not route:
                continue

            xs = [self.cust[i]['xcoord'] for i in route]
            ys = [self.cust[i]['ycoord'] for i in route]
            ds = [self.cust[i]['demand'] for i in route]
            rn = len(route)

            # 使用需求加权质心表示聚合节点的空间位置
            demand_sum = max(1e-6, float(sum(ds)))
            weights = [d / demand_sum for d in ds]
            centroid_x = float(sum(w * x for w, x in zip(weights, xs)))
            centroid_y = float(sum(w * y for w, y in zip(weights, ys)))

            # 计算内部服务时间（访问聚合节点内部所有客户的服务+行驶时间）
            service_time = 0.0
            for idx, customer_idx in enumerate(route):
                service_time += float(self.cust[customer_idx]['service_time'])
                if idx < rn - 1:
                    nxt = route[idx + 1]
                    service_time += float(self.D[customer_idx + 1, nxt + 1])

            # 计算前向累计时间，用于估计可行的出发时间窗口
            prefix_travel = []
            cumulative = float(self.D[0, route[0] + 1])
            prefix_travel.append(cumulative)
            for idx in range(1, rn):
                prev = route[idx - 1]
                curr = route[idx]
                cumulative += float(self.cust[prev]['service_time'])
                cumulative += float(self.D[prev + 1, curr + 1])
                prefix_travel.append(cumulative)

            ready_candidates = []
            due_candidates = []
            for offset, customer_idx in zip(prefix_travel, route):
                ready = float(self.cust[customer_idx]['ready_time'])
                due = float(self.cust[customer_idx]['due_date'])
                ready_candidates.append(ready - offset)
                due_candidates.append(due - offset)

            ready_time = max(0.0, max(ready_candidates)) if ready_candidates else 0.0
            due_time = min(due_candidates) if due_candidates else ready_time + service_time
            if due_time < ready_time:
                due_time = ready_time + max(1.0, service_time)

            deepdata['customer'].append({
                'x': centroid_x,
                'y': centroid_y,
                'demand': demand_sum,
                'ServiceTime': service_time,
                'ReadyTime': ready_time,
                'DueDate': due_time
            })

        return deepdata


# === 附加工具函数：基于路径的自适应降维 ===
def adaptive_route_reduction(routes, instance, distance_threshold=5.0, verbose=False):
    """
    基于路径内相邻客户距离的自适应降维方法
    :param routes: 解码后的路径列表，每个路径包含 'customers' 字段
    :param instance: VRPTWInstance 实例
    :param distance_threshold: 距离阈值，相邻客户距离小于等于此值时合并
    :param verbose: 是否打印详细信息
    :return: 降维后的客户分组列表
    """
    coords = instance.ordinary_customers

    def euclidean_distance(u, v):
        cu, cv = coords[u], coords[v]
        dx = cu['xcoord'] - cv['xcoord']
        dy = cu['ycoord'] - cv['ycoord']
        return np.sqrt(dx*dx + dy*dy)

    reduced_groups = []
    for route_idx, route in enumerate(routes):
        customers = route.get('customers', [])
        if not customers:
            continue

        distances = [euclidean_distance(customers[i], customers[i+1])
                     for i in range(len(customers)-1)]

        current_group = [customers[0]]
        route_groups = []
        for i in range(len(customers)-1):
            if distances[i] <= distance_threshold:
                current_group.append(customers[i+1])
            else:
                route_groups.append(current_group.copy())
                current_group = [customers[i+1]]
        if current_group:
            route_groups.append(current_group)

        reduced_groups.extend(route_groups)

    return reduced_groups


# === 附加工具函数：可视化降维前后路径对比 ===
def visualize_reduction_comparison(original_routes, reduced_groups, instance, title="路径降维对比"):
    try:
        import matplotlib.pyplot as plt

        depot = instance.warehouse
        customers = instance.ordinary_customers
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 原始路径
        ax1.scatter(depot['xcoord'], depot['ycoord'], c='red', marker='s', s=120, label='仓库')
        ax1.scatter([c['xcoord'] for c in customers],
                    [c['ycoord'] for c in customers],
                    c='lightgray', s=30, label='客户')
        colors = plt.cm.tab10(np.linspace(0, 1, len(original_routes)))
        for i, route in enumerate(original_routes):
            route_customers = route.get('customers', [])
            if not route_customers:
                continue
            path_x = [depot['xcoord']] + [customers[c]['xcoord'] for c in route_customers] + [depot['xcoord']]
            path_y = [depot['ycoord']] + [customers[c]['ycoord'] for c in route_customers] + [depot['ycoord']]
            ax1.plot(path_x, path_y, c=colors[i], linewidth=2, alpha=0.7, label=f'路径{i+1}')
        ax1.set_title('原始路径')

        # 降维后
        ax2.scatter(depot['xcoord'], depot['ycoord'], c='red', marker='s', s=120, label='仓库')
        ax2.scatter([c['xcoord'] for c in customers],
                    [c['ycoord'] for c in customers],
                    c='lightgray', s=30, label='客户')
        group_colors = plt.cm.Set3(np.linspace(0, 1, len(reduced_groups)))
        for i, group in enumerate(reduced_groups):
            if not group:
                continue
            group_x = [customers[c]['xcoord'] for c in group]
            group_y = [customers[c]['ycoord'] for c in group]
            ax2.scatter(group_x, group_y, c=[group_colors[i]], s=50, alpha=0.8, label=f'分组{i+1}')
            if len(group) > 1:
                for j in range(len(group)-1):
                    x1, y1 = customers[group[j]]['xcoord'], customers[group[j]]['ycoord']
                    x2, y2 = customers[group[j+1]]['xcoord'], customers[group[j+1]]['ycoord']
                    ax2.plot([x1, x2], [y1, y2], c=group_colors[i], linewidth=3, alpha=0.8)
        ax2.set_title('降维后分组')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    except ImportError:
        print("matplotlib 未安装，无法显示可视化图表")


# === Deep 空间 VRPTW 实例类 & 映射函数 ===
class DeepVRPTWInstance:
    def __init__(self, deepdata):
        self.vehicle_info = deepdata['vehicle']
        self.warehouse = {
            'cust_no': 0,
            'xcoord': deepdata['depot']['xcoord'],
            'ycoord': deepdata['depot']['ycoord'],
            'demand': 0,
            'ready_time': 0,
            'due_date': deepdata['depot']['due_date'],
            'service_time': 0
        }
        self.ordinary_customers = []
        for idx, c in enumerate(deepdata['customer']):
            self.ordinary_customers.append({
                'cust_no': idx + 1,
                'xcoord': c['x'],
                'ycoord': c['y'],
                'demand': c['demand'],
                'ready_time': c['ReadyTime'],
                'due_date': c['DueDate'],
                'service_time': c['ServiceTime']
            })

        nodes = [self.warehouse] + self.ordinary_customers
        n = len(nodes)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dx = nodes[i]['xcoord'] - nodes[j]['xcoord']
                dy = nodes[i]['ycoord'] - nodes[j]['ycoord']
                D[i, j] = np.hypot(dx, dy)
        self.distance_matrix = D


def map_deep_to_full(deep_solution, vc_new):
    full_seq = []
    for d_idx in deep_solution:
        full_seq.extend(vc_new[d_idx])
    return full_seq


def map_full_to_deep_by_label(full_seq, cluster_label, Nd):
    seen = set()
    deep_seq = []
    for node in full_seq:
        cl = cluster_label[node]
        if cl not in seen:
            seen.add(cl)
            deep_seq.append(cl)
        if len(seen) == Nd:
            break
    if len(deep_seq) < Nd:
        for cl in range(Nd):
            if cl not in seen:
                deep_seq.append(cl)
    return deep_seq
