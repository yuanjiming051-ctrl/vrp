import random
import math

class InitialSolutionGenerator:
    """
    多种启发式方法生成高质量VRPTW初始解
    """

    def __init__(self, instance):
        """
        :param instance: VRPTWInstance 对象，应包含 ordinary_customers、vehicle_info 等属性
        """
        self.instance = instance

    def generate_random_solution(self):
        """
        生成多种启发式初始解，返回最佳的一个
        """
        solutions = []
        
        # 方法1：时间窗优先策略
        solutions.append(self._time_window_first())
        
        # 方法2：最近邻策略
        solutions.append(self._nearest_neighbor())
        
        # 方法3：节约算法策略
        solutions.append(self._savings_algorithm())
        
        # 方法4：混合策略（时间窗+地理位置）
        solutions.append(self._hybrid_strategy())
        
        # 方法5：随机策略（保持多样性）
        solutions.append(self._random_strategy())
        
        # 评估所有解，返回最佳的
        return self._select_best_solution(solutions)

    def _time_window_first(self):
        """时间窗优先策略：按ready_time排序"""
        M = len(self.instance.ordinary_customers)
        customers_with_time = [(i, self.instance.ordinary_customers[i]['ready_time']) 
                              for i in range(M)]
        customers_with_time.sort(key=lambda x: x[1])
        return [customer[0] for customer in customers_with_time]

    def _nearest_neighbor(self):
        """最近邻策略：从depot开始，每次选择最近的未访问客户"""
        M = len(self.instance.ordinary_customers)
        unvisited = set(range(M))
        solution = []
        depot = self.instance.warehouse
        
        # 从距离depot最近的客户开始
        current_pos = depot
        
        while unvisited:
            min_distance = float('inf')
            nearest_customer = None
            
            for customer_idx in unvisited:
                customer = self.instance.ordinary_customers[customer_idx]
                distance = self._calculate_distance(current_pos, customer)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_customer = customer_idx
            
            solution.append(nearest_customer)
            unvisited.remove(nearest_customer)
            current_pos = self.instance.ordinary_customers[nearest_customer]
        
        return solution

    def _savings_algorithm(self):
        """节约算法策略：基于节约值构建路径"""
        M = len(self.instance.ordinary_customers)
        depot = self.instance.warehouse
        
        # 计算所有客户对的节约值
        savings = []
        for i in range(M):
            for j in range(i + 1, M):
                customer_i = self.instance.ordinary_customers[i]
                customer_j = self.instance.ordinary_customers[j]
                
                dist_depot_i = self._calculate_distance(depot, customer_i)
                dist_depot_j = self._calculate_distance(depot, customer_j)
                dist_i_j = self._calculate_distance(customer_i, customer_j)
                
                # 节约值 = 从depot到i的距离 + 从depot到j的距离 - 从i到j的距离
                saving = dist_depot_i + dist_depot_j - dist_i_j
                savings.append((saving, i, j))
        
        # 按节约值降序排序
        savings.sort(reverse=True)
        
        # 构建解
        solution = list(range(M))
        
        # 基于节约值调整顺序
        for saving, i, j in savings[:M//2]:  # 只使用前一半的节约值
            if abs(solution.index(i) - solution.index(j)) > 1:
                # 尝试将i和j放在相邻位置
                idx_i = solution.index(i)
                idx_j = solution.index(j)
                if idx_i > idx_j:
                    idx_i, idx_j = idx_j, idx_i
                    i, j = j, i
                
                solution.remove(j)
                solution.insert(idx_i + 1, j)
        
        return solution

    def _hybrid_strategy(self):
        """混合策略：结合时间窗和地理位置"""
        M = len(self.instance.ordinary_customers)
        depot = self.instance.warehouse
        
        # 计算每个客户的综合评分
        customer_scores = []
        for i in range(M):
            customer = self.instance.ordinary_customers[i]
            
            # 时间窗紧迫度（ready_time越小越紧急）
            time_urgency = 1.0 / (customer['ready_time'] + 1)
            
            # 距离因子（距离depot越近越好）
            distance = self._calculate_distance(depot, customer)
            distance_factor = 1.0 / (distance + 1)
            
            # 时间窗宽度（due_date - ready_time越小越紧急）
            time_window_width = customer['due_date'] - customer['ready_time']
            window_urgency = 1.0 / (time_window_width + 1)
            
            # 综合评分（权重可调）
            score = 0.4 * time_urgency + 0.3 * distance_factor + 0.3 * window_urgency
            customer_scores.append((score, i))
        
        # 按综合评分降序排序
        customer_scores.sort(reverse=True)
        return [customer[1] for customer in customer_scores]

    def _random_strategy(self):
        """随机策略：保持种群多样性"""
        M = len(self.instance.ordinary_customers)
        solution = list(range(M))
        random.shuffle(solution)
        return solution

    def _calculate_distance(self, pos1, pos2):
        """计算两点间的欧几里得距离"""
        return math.sqrt((pos1['xcoord'] - pos2['xcoord'])**2 + 
                        (pos1['ycoord'] - pos2['ycoord'])**2)

    def _select_best_solution(self, solutions):
        """从多个解中选择最佳的一个"""
        # 这里简化处理，实际应该使用解码器评估
        # 暂时返回时间窗优先策略的结果
        return solutions[0]  # 返回时间窗优先策略


if __name__ == "__main__":
    from initial_solution import InitialSolutionGenerator
    class DummyInstance:
        def __init__(self):
            self.ordinary_customers = list(range(5))
    inst = DummyInstance()
    gen = InitialSolutionGenerator(inst)
    sol = gen.generate_random_solution()
    print("InitialSolutionGenerator output:", sol)
