#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门的Deep空间进化算法
为Deep空间设计的独立进化算法，包含遗传算法、模拟退火和混合进化策略
"""

import numpy as np
import random
import math
import copy
from typing import List, Dict, Tuple, Any, Optional, Callable
from collections import defaultdict
import heapq
from abc import ABC, abstractmethod


class DeepSpaceIndividual:
    """Deep空间个体"""
    
    def __init__(self, chromosome: List[List[int]], deep_nodes_count: int):
        """
        初始化个体
        
        Args:
            chromosome: 染色体（路径列表）
            deep_nodes_count: Deep节点总数
        """
        self.chromosome = chromosome
        self.deep_nodes_count = deep_nodes_count
        self.fitness = float('inf')
        self.feasible = False
        self.evaluation_cache = {}
        
        # 个体特征
        self.age = 0
        self.diversity_score = 0.0
        self.improvement_history = []
    
    def copy(self):
        """复制个体"""
        new_individual = DeepSpaceIndividual(
            copy.deepcopy(self.chromosome), 
            self.deep_nodes_count
        )
        new_individual.fitness = self.fitness
        new_individual.feasible = self.feasible
        new_individual.age = self.age
        new_individual.diversity_score = self.diversity_score
        return new_individual
    
    def get_used_nodes(self) -> set:
        """获取使用的节点集合"""
        used_nodes = set()
        for route in self.chromosome:
            used_nodes.update(route)
        return used_nodes
    
    def get_route_count(self) -> int:
        """获取路径数量"""
        return len([route for route in self.chromosome if route])
    
    def calculate_diversity(self, other: 'DeepSpaceIndividual') -> float:
        """计算与另一个个体的多样性"""
        # 基于路径结构的多样性
        self_routes = set(tuple(route) for route in self.chromosome if route)
        other_routes = set(tuple(route) for route in other.chromosome if route)
        
        if not self_routes and not other_routes:
            return 0.0
        
        intersection = len(self_routes & other_routes)
        union = len(self_routes | other_routes)
        
        return 1.0 - (intersection / union if union > 0 else 0.0)
    
    def __lt__(self, other):
        """比较操作符，用于排序"""
        if self.feasible != other.feasible:
            return self.feasible > other.feasible  # 可行解优先
        return self.fitness < other.fitness


class DeepSpaceEvolutionStrategy(ABC):
    """Deep空间进化策略抽象基类"""
    
    @abstractmethod
    def evolve_population(self, population: List[DeepSpaceIndividual], 
                         generation: int) -> List[DeepSpaceIndividual]:
        """进化种群"""
        pass


class DeepSpaceGeneticAlgorithm(DeepSpaceEvolutionStrategy):
    """Deep空间遗传算法"""
    
    def __init__(self, crossover_rate: float = 0.8, mutation_rate: float = 0.2,
                 elite_size: int = 2, tournament_size: int = 3):
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
    
    def evolve_population(self, population: List[DeepSpaceIndividual], 
                         generation: int) -> List[DeepSpaceIndividual]:
        """遗传算法进化"""
        new_population = []
        
        # 精英保留
        population.sort()
        elites = population[:self.elite_size]
        new_population.extend([elite.copy() for elite in elites])
        
        # 生成新个体
        while len(new_population) < len(population):
            # 选择父母
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            # 交叉
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # 变异
            if random.random() < self.mutation_rate:
                child1 = self._mutate(child1)
            if random.random() < self.mutation_rate:
                child2 = self._mutate(child2)
            
            new_population.extend([child1, child2])
        
        return new_population[:len(population)]
    
    def _tournament_selection(self, population: List[DeepSpaceIndividual]) -> DeepSpaceIndividual:
        """锦标赛选择"""
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return min(tournament)
    
    def _crossover(self, parent1: DeepSpaceIndividual, 
                  parent2: DeepSpaceIndividual) -> Tuple[DeepSpaceIndividual, DeepSpaceIndividual]:
        """交叉操作"""
        # 使用路径级别的交叉
        child1_routes = []
        child2_routes = []
        
        # 收集所有节点
        all_nodes = set()
        for route in parent1.chromosome + parent2.chromosome:
            all_nodes.update(route)
        
        used_nodes1 = set()
        used_nodes2 = set()
        
        # 随机选择路径进行交叉
        max_routes = max(len(parent1.chromosome), len(parent2.chromosome))
        
        for i in range(max_routes):
            if random.random() < 0.5:
                # 从parent1选择路径
                if i < len(parent1.chromosome):
                    route = [node for node in parent1.chromosome[i] if node not in used_nodes1]
                    if route:
                        child1_routes.append(route)
                        used_nodes1.update(route)
                
                if i < len(parent2.chromosome):
                    route = [node for node in parent2.chromosome[i] if node not in used_nodes2]
                    if route:
                        child2_routes.append(route)
                        used_nodes2.update(route)
            else:
                # 从parent2选择路径
                if i < len(parent2.chromosome):
                    route = [node for node in parent2.chromosome[i] if node not in used_nodes1]
                    if route:
                        child1_routes.append(route)
                        used_nodes1.update(route)
                
                if i < len(parent1.chromosome):
                    route = [node for node in parent1.chromosome[i] if node not in used_nodes2]
                    if route:
                        child2_routes.append(route)
                        used_nodes2.update(route)
        
        # 添加未使用的节点
        unused_nodes1 = all_nodes - used_nodes1
        unused_nodes2 = all_nodes - used_nodes2
        
        if unused_nodes1:
            child1_routes.append(list(unused_nodes1))
        if unused_nodes2:
            child2_routes.append(list(unused_nodes2))
        
        child1 = DeepSpaceIndividual(child1_routes, parent1.deep_nodes_count)
        child2 = DeepSpaceIndividual(child2_routes, parent2.deep_nodes_count)
        
        return child1, child2
    
    def _mutate(self, individual: DeepSpaceIndividual) -> DeepSpaceIndividual:
        """变异操作"""
        mutated = individual.copy()
        
        if not mutated.chromosome:
            return mutated
        
        mutation_type = random.choice(['swap_nodes', 'move_node', 'split_route', 'merge_routes'])
        
        if mutation_type == 'swap_nodes':
            self._swap_nodes_mutation(mutated)
        elif mutation_type == 'move_node':
            self._move_node_mutation(mutated)
        elif mutation_type == 'split_route':
            self._split_route_mutation(mutated)
        elif mutation_type == 'merge_routes':
            self._merge_routes_mutation(mutated)
        
        return mutated
    
    def _swap_nodes_mutation(self, individual: DeepSpaceIndividual):
        """交换节点变异"""
        all_nodes = []
        node_positions = {}
        
        for route_idx, route in enumerate(individual.chromosome):
            for pos, node in enumerate(route):
                all_nodes.append(node)
                node_positions[node] = (route_idx, pos)
        
        if len(all_nodes) < 2:
            return
        
        # 随机选择两个节点交换
        node1, node2 = random.sample(all_nodes, 2)
        route1_idx, pos1 = node_positions[node1]
        route2_idx, pos2 = node_positions[node2]
        
        # 执行交换
        individual.chromosome[route1_idx][pos1] = node2
        individual.chromosome[route2_idx][pos2] = node1
    
    def _move_node_mutation(self, individual: DeepSpaceIndividual):
        """移动节点变异"""
        non_empty_routes = [i for i, route in enumerate(individual.chromosome) if route]
        
        if len(non_empty_routes) < 1:
            return
        
        # 选择源路径
        source_route_idx = random.choice(non_empty_routes)
        source_route = individual.chromosome[source_route_idx]
        
        if not source_route:
            return
        
        # 选择要移动的节点
        node_pos = random.randint(0, len(source_route) - 1)
        node = source_route.pop(node_pos)
        
        # 选择目标路径
        if random.random() < 0.3 and len(individual.chromosome) < 10:  # 创建新路径
            individual.chromosome.append([node])
        else:
            # 移动到现有路径
            target_route_idx = random.randint(0, len(individual.chromosome) - 1)
            target_route = individual.chromosome[target_route_idx]
            insert_pos = random.randint(0, len(target_route))
            target_route.insert(insert_pos, node)
        
        # 清理空路径
        individual.chromosome = [route for route in individual.chromosome if route]
    
    def _split_route_mutation(self, individual: DeepSpaceIndividual):
        """分割路径变异"""
        non_empty_routes = [i for i, route in enumerate(individual.chromosome) if len(route) > 1]
        
        if not non_empty_routes:
            return
        
        route_idx = random.choice(non_empty_routes)
        route = individual.chromosome[route_idx]
        
        # 随机选择分割点
        split_point = random.randint(1, len(route) - 1)
        
        # 分割路径
        route1 = route[:split_point]
        route2 = route[split_point:]
        
        individual.chromosome[route_idx] = route1
        individual.chromosome.append(route2)
    
    def _merge_routes_mutation(self, individual: DeepSpaceIndividual):
        """合并路径变异"""
        non_empty_routes = [i for i, route in enumerate(individual.chromosome) if route]
        
        if len(non_empty_routes) < 2:
            return
        
        # 随机选择两条路径合并
        route1_idx, route2_idx = random.sample(non_empty_routes, 2)
        
        # 合并路径
        merged_route = individual.chromosome[route1_idx] + individual.chromosome[route2_idx]
        
        # 更新染色体
        individual.chromosome[route1_idx] = merged_route
        individual.chromosome.pop(route2_idx)


class DeepSpaceSimulatedAnnealing(DeepSpaceEvolutionStrategy):
    """Deep空间模拟退火算法"""
    
    def __init__(self, initial_temp: float = 1000.0, cooling_rate: float = 0.95,
                 min_temp: float = 1.0, iterations_per_temp: int = 10):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.iterations_per_temp = iterations_per_temp
    
    def evolve_population(self, population: List[DeepSpaceIndividual], 
                         generation: int) -> List[DeepSpaceIndividual]:
        """模拟退火进化"""
        # 对每个个体应用模拟退火
        improved_population = []
        
        for individual in population:
            improved = self._simulated_annealing(individual, generation)
            improved_population.append(improved)
        
        return improved_population
    
    def _simulated_annealing(self, individual: DeepSpaceIndividual, 
                           generation: int) -> DeepSpaceIndividual:
        """对单个个体应用模拟退火"""
        current = individual.copy()
        best = current.copy()
        
        # 计算当前温度
        temp = self.initial_temp * (self.cooling_rate ** generation)
        temp = max(temp, self.min_temp)
        
        for _ in range(self.iterations_per_temp):
            # 生成邻域解
            neighbor = self._generate_neighbor(current)
            
            # 计算接受概率
            if neighbor.fitness < current.fitness:
                current = neighbor
                if neighbor.fitness < best.fitness:
                    best = neighbor
            else:
                delta = neighbor.fitness - current.fitness
                probability = math.exp(-delta / temp) if temp > 0 else 0
                if random.random() < probability:
                    current = neighbor
        
        return best
    
    def _generate_neighbor(self, individual: DeepSpaceIndividual) -> DeepSpaceIndividual:
        """生成邻域解"""
        neighbor = individual.copy()
        
        # 随机选择邻域操作
        operations = ['2opt', 'relocate', 'swap', 'or_opt']
        operation = random.choice(operations)
        
        if operation == '2opt':
            self._apply_2opt(neighbor)
        elif operation == 'relocate':
            self._apply_relocate(neighbor)
        elif operation == 'swap':
            self._apply_swap(neighbor)
        elif operation == 'or_opt':
            self._apply_or_opt(neighbor)
        
        return neighbor
    
    def _apply_2opt(self, individual: DeepSpaceIndividual):
        """应用2-opt操作"""
        if not individual.chromosome:
            return
        
        route_idx = random.randint(0, len(individual.chromosome) - 1)
        route = individual.chromosome[route_idx]
        
        if len(route) < 4:
            return
        
        i = random.randint(0, len(route) - 2)
        j = random.randint(i + 2, len(route) - 1)
        
        # 执行2-opt
        individual.chromosome[route_idx] = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
    
    def _apply_relocate(self, individual: DeepSpaceIndividual):
        """应用relocate操作"""
        if not individual.chromosome:
            return
        
        # 选择源路径和节点
        non_empty_routes = [i for i, route in enumerate(individual.chromosome) if route]
        if not non_empty_routes:
            return
        
        source_route_idx = random.choice(non_empty_routes)
        source_route = individual.chromosome[source_route_idx]
        
        if not source_route:
            return
        
        node_idx = random.randint(0, len(source_route) - 1)
        node = source_route.pop(node_idx)
        
        # 选择目标路径和位置
        target_route_idx = random.randint(0, len(individual.chromosome) - 1)
        target_route = individual.chromosome[target_route_idx]
        insert_pos = random.randint(0, len(target_route))
        target_route.insert(insert_pos, node)
        
        # 清理空路径
        individual.chromosome = [route for route in individual.chromosome if route]
    
    def _apply_swap(self, individual: DeepSpaceIndividual):
        """应用swap操作"""
        all_nodes = []
        node_positions = {}
        
        for route_idx, route in enumerate(individual.chromosome):
            for pos, node in enumerate(route):
                all_nodes.append(node)
                node_positions[node] = (route_idx, pos)
        
        if len(all_nodes) < 2:
            return
        
        node1, node2 = random.sample(all_nodes, 2)
        route1_idx, pos1 = node_positions[node1]
        route2_idx, pos2 = node_positions[node2]
        
        individual.chromosome[route1_idx][pos1] = node2
        individual.chromosome[route2_idx][pos2] = node1
    
    def _apply_or_opt(self, individual: DeepSpaceIndividual):
        """应用Or-opt操作"""
        if not individual.chromosome:
            return
        
        non_empty_routes = [i for i, route in enumerate(individual.chromosome) if len(route) >= 2]
        if not non_empty_routes:
            return
        
        route_idx = random.choice(non_empty_routes)
        route = individual.chromosome[route_idx]
        
        # 选择要移动的子序列长度（1-3）
        subseq_len = random.randint(1, min(3, len(route)))
        start_pos = random.randint(0, len(route) - subseq_len)
        
        # 提取子序列
        subseq = route[start_pos:start_pos + subseq_len]
        remaining_route = route[:start_pos] + route[start_pos + subseq_len:]
        
        # 重新插入子序列
        insert_pos = random.randint(0, len(remaining_route))
        new_route = remaining_route[:insert_pos] + subseq + remaining_route[insert_pos:]
        
        individual.chromosome[route_idx] = new_route


class DeepSpaceHybridEvolution(DeepSpaceEvolutionStrategy):
    """Deep空间混合进化算法"""
    
    def __init__(self, ga_weight: float = 0.6, sa_weight: float = 0.4):
        self.ga = DeepSpaceGeneticAlgorithm()
        self.sa = DeepSpaceSimulatedAnnealing()
        self.ga_weight = ga_weight
        self.sa_weight = sa_weight
    
    def evolve_population(self, population: List[DeepSpaceIndividual], 
                         generation: int) -> List[DeepSpaceIndividual]:
        """混合进化"""
        # 分割种群
        ga_size = int(len(population) * self.ga_weight)
        sa_size = len(population) - ga_size
        
        # 选择不同的个体进行不同的进化
        population.sort()
        ga_population = population[:ga_size]
        sa_population = population[ga_size:ga_size + sa_size]
        
        # 应用不同的进化策略
        evolved_ga = self.ga.evolve_population(ga_population, generation)
        evolved_sa = self.sa.evolve_population(sa_population, generation)
        
        # 合并结果
        combined_population = evolved_ga + evolved_sa
        
        # 选择最优个体
        combined_population.sort()
        return combined_population[:len(population)]


class DeepSpaceEvolutionEngine:
    """Deep空间进化引擎"""
    
    def __init__(self, enhanced_constructor, decoder, population_size: int = 50,
                 max_generations: int = 100, strategy: str = 'hybrid'):
        """
        初始化进化引擎
        
        Args:
            enhanced_constructor: 增强的Deep空间构造器
            decoder: Deep空间解码器
            population_size: 种群大小
            max_generations: 最大代数
            strategy: 进化策略 ('genetic', 'simulated_annealing', 'hybrid')
        """
        self.constructor = enhanced_constructor
        self.decoder = decoder
        self.population_size = population_size
        self.max_generations = max_generations
        
        # 选择进化策略
        if strategy == 'genetic':
            self.evolution_strategy = DeepSpaceGeneticAlgorithm()
        elif strategy == 'simulated_annealing':
            self.evolution_strategy = DeepSpaceSimulatedAnnealing()
        elif strategy == 'hybrid':
            self.evolution_strategy = DeepSpaceHybridEvolution()
        else:
            raise ValueError(f"未知的进化策略: {strategy}")
        
        # 进化统计
        self.evolution_history = []
        self.best_individual = None
        self.diversity_history = []
    
    def initialize_population(self) -> List[DeepSpaceIndividual]:
        """初始化种群"""
        population = []
        deep_nodes_count = len(self.constructor.deep_nodes)
        
        for _ in range(self.population_size):
            # 生成随机个体
            individual = self._generate_random_individual(deep_nodes_count)
            population.append(individual)
        
        return population
    
    def _generate_random_individual(self, deep_nodes_count: int) -> DeepSpaceIndividual:
        """生成随机个体"""
        # 随机生成路径数量
        num_routes = random.randint(1, min(deep_nodes_count, 8))
        
        # 随机分配节点到路径
        nodes = list(range(deep_nodes_count))
        random.shuffle(nodes)
        
        chromosome = []
        nodes_per_route = len(nodes) // num_routes
        
        for i in range(num_routes):
            start_idx = i * nodes_per_route
            if i == num_routes - 1:  # 最后一条路径包含剩余所有节点
                route = nodes[start_idx:]
            else:
                route = nodes[start_idx:start_idx + nodes_per_route]
            
            if route:
                chromosome.append(route)
        
        return DeepSpaceIndividual(chromosome, deep_nodes_count)
    
    def evaluate_individual(self, individual: DeepSpaceIndividual) -> float:
        """评估个体适应度"""
        # 解码为原始空间解
        try:
            original_solution = self.decoder.decode_deep_solution(
                individual.chromosome, strategy='hybrid')
            
            # 评估解质量
            evaluation = self.decoder.evaluate_solution(original_solution)
            
            individual.fitness = evaluation['objective']
            individual.feasible = evaluation['feasible']
            
            return individual.fitness
        except Exception as e:
            # 解码失败，设置为最差适应度
            individual.fitness = float('inf')
            individual.feasible = False
            return individual.fitness
    
    def evaluate_population(self, population: List[DeepSpaceIndividual]):
        """评估整个种群"""
        for individual in population:
            self.evaluate_individual(individual)
    
    def calculate_population_diversity(self, population: List[DeepSpaceIndividual]) -> float:
        """计算种群多样性"""
        if len(population) < 2:
            return 0.0
        
        total_diversity = 0.0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                diversity = population[i].calculate_diversity(population[j])
                total_diversity += diversity
                comparisons += 1
        
        return total_diversity / comparisons if comparisons > 0 else 0.0
    
    def evolve(self) -> DeepSpaceIndividual:
        """执行进化过程"""
        # 初始化种群
        population = self.initialize_population()
        self.evaluate_population(population)
        
        # 记录初始最优个体
        population.sort()
        self.best_individual = population[0].copy()
        
        print(f"初始最优适应度: {self.best_individual.fitness}")
        
        # 进化循环
        for generation in range(self.max_generations):
            # 进化种群
            population = self.evolution_strategy.evolve_population(population, generation)
            
            # 评估新种群
            self.evaluate_population(population)
            
            # 更新最优个体
            population.sort()
            if population[0].fitness < self.best_individual.fitness:
                self.best_individual = population[0].copy()
                print(f"代数 {generation}: 新的最优适应度 {self.best_individual.fitness}")
            
            # 计算种群多样性
            diversity = self.calculate_population_diversity(population)
            self.diversity_history.append(diversity)
            
            # 记录进化历史
            generation_stats = {
                'generation': generation,
                'best_fitness': population[0].fitness,
                'avg_fitness': np.mean([ind.fitness for ind in population if ind.fitness != float('inf')]),
                'diversity': diversity,
                'feasible_count': sum(1 for ind in population if ind.feasible)
            }
            self.evolution_history.append(generation_stats)
            
            # 早停条件
            if generation > 20 and self._should_early_stop():
                print(f"早停于代数 {generation}")
                break
            
            # 多样性维护
            if diversity < 0.1 and generation % 10 == 0:
                population = self._maintain_diversity(population)
        
        return self.best_individual
    
    def _should_early_stop(self) -> bool:
        """判断是否应该早停"""
        if len(self.evolution_history) < 20:
            return False
        
        # 检查最近20代的改进
        recent_best = [stats['best_fitness'] for stats in self.evolution_history[-20:]]
        improvement = recent_best[0] - recent_best[-1]
        
        return improvement < 0.001  # 改进很小时早停
    
    def _maintain_diversity(self, population: List[DeepSpaceIndividual]) -> List[DeepSpaceIndividual]:
        """维护种群多样性"""
        # 保留最优个体
        population.sort()
        elite_size = max(1, len(population) // 10)
        elites = population[:elite_size]
        
        # 重新生成部分个体
        new_individuals = []
        for _ in range(len(population) - elite_size):
            new_individual = self._generate_random_individual(self.constructor.deep_nodes.__len__())
            new_individuals.append(new_individual)
        
        return elites + new_individuals
    
    def get_best_solution(self) -> Tuple[List[List[int]], Dict[str, Any]]:
        """获取最优解"""
        if self.best_individual is None:
            raise ValueError("请先执行进化过程")
        
        # 解码最优个体
        original_solution = self.decoder.decode_deep_solution(
            self.best_individual.chromosome, strategy='hybrid')
        
        # 改进解
        improved_solution = self.decoder.improve_solution(original_solution)
        
        # 评估最终解
        evaluation = self.decoder.evaluate_solution(improved_solution)
        
        return improved_solution, evaluation
    
    def plot_evolution_history(self, save_path: str = None):
        """绘制进化历史"""
        if not self.evolution_history:
            print("没有进化历史数据")
            return
        
        import matplotlib.pyplot as plt
        
        generations = [stats['generation'] for stats in self.evolution_history]
        best_fitness = [stats['best_fitness'] for stats in self.evolution_history]
        avg_fitness = [stats['avg_fitness'] for stats in self.evolution_history]
        diversity = [stats['diversity'] for stats in self.evolution_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 适应度曲线
        ax1.plot(generations, best_fitness, 'b-', label='最优适应度', linewidth=2)
        ax1.plot(generations, avg_fitness, 'r--', label='平均适应度', alpha=0.7)
        ax1.set_xlabel('代数')
        ax1.set_ylabel('适应度')
        ax1.set_title('Deep空间进化算法 - 适应度变化')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 多样性曲线
        ax2.plot(generations, diversity, 'g-', label='种群多样性', linewidth=2)
        ax2.set_xlabel('代数')
        ax2.set_ylabel('多样性')
        ax2.set_title('种群多样性变化')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# 使用示例
if __name__ == "__main__":
    from vrptw_instance import VRPTWInstance
    from enhanced_deep_constructor import EnhancedDeepConstructor
    from deep_space_decoder import DeepSpaceDecoder
    
    # 加载实例
    instance = VRPTWInstance("data/RC1_2_1.txt")
    
    # 创建Deep空间构造器
    constructor = EnhancedDeepConstructor(instance, cluster_method='time_spatial')
    deep_nodes = constructor.construct_deep_space(target_clusters=8)
    
    # 创建解码器
    decoder = DeepSpaceDecoder(constructor, instance)
    
    # 创建进化引擎
    evolution_engine = DeepSpaceEvolutionEngine(
        constructor, decoder, 
        population_size=30, 
        max_generations=50, 
        strategy='hybrid'
    )
    
    # 执行进化
    print("开始Deep空间进化...")
    best_individual = evolution_engine.evolve()
    
    # 获取最优解
    best_solution, evaluation = evolution_engine.get_best_solution()
    
    print(f"最优解评估: {evaluation}")
    print(f"Deep空间解: {best_individual.chromosome}")
    print(f"原始空间解: {best_solution}")
    
    # 绘制进化历史
    evolution_engine.plot_evolution_history()