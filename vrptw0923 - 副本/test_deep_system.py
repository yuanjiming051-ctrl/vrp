#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep空间独立系统测试脚本
测试增强的Deep空间构造器、解码器、进化算法和状态编码器的集成功能
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import json
from typing import List, Dict, Any, Tuple
import os
import sys

# 导入必要的模块
from vrptw_instance import VRPTWInstance
from enhanced_deep_constructor import EnhancedDeepConstructor
from deep_space_decoder import DeepSpaceDecoder
from deep_space_evolution import DeepSpaceEvolutionEngine, DeepSpaceGeneticAlgorithm
from deep_space_encoder import DeepSpaceStateEncoder


class DeepSystemTester:
    """Deep空间独立系统测试器"""
    
    def __init__(self, instance_file: str):
        """
        初始化测试器
        
        Args:
            instance_file: VRPTW实例文件路径
        """
        self.instance_file = instance_file
        self.instance = None
        self.constructor = None
        self.decoder = None
        self.evolution_engine = None
        self.encoder = None
        
        # 测试结果
        self.test_results = {}
        
    def setup_system(self):
        """设置Deep空间独立系统"""
        print("=== 设置Deep空间独立系统 ===")
        
        # 1. 加载VRPTW实例
        print(f"加载实例: {self.instance_file}")
        self.instance = VRPTWInstance(self.instance_file)
        print(f"客户数量: {self.instance.customer_count}")
        print(f"车辆容量: {self.instance.vehicle_capacity}")
        
        # 2. 创建增强的Deep空间构造器
        print("\n创建增强的Deep空间构造器...")
        self.constructor = EnhancedDeepConstructor(
            self.instance, 
            cluster_method='time_spatial'
        )
        
        # 3. 构造Deep空间
        target_clusters = min(8, self.instance.customer_count // 3)
        print(f"构造Deep空间，目标聚类数: {target_clusters}")
        deep_nodes = self.constructor.construct_deep_space(target_clusters=target_clusters)
        print(f"Deep节点数量: {len(deep_nodes)}")
        
        # 4. 创建解码器
        print("\n创建Deep空间解码器...")
        self.decoder = DeepSpaceDecoder(self.constructor)
        
        # 5. 创建进化算法引擎
        print("\n创建Deep空间进化算法引擎...")
        ga_strategy = DeepSpaceGeneticAlgorithm(
            population_size=20,
            crossover_rate=0.8,
            mutation_rate=0.2
        )
        
        self.evolution_engine = DeepSpaceEvolutionEngine(
            deep_constructor=self.constructor,
            decoder=self.decoder,
            evolution_strategy=ga_strategy
        )
        
        # 6. 创建状态编码器
        print("\n创建Deep空间状态编码器...")
        self.encoder = DeepSpaceStateEncoder(
            self.constructor,
            feature_dim=128,
            use_normalization=True
        )
        
        print("Deep空间独立系统设置完成！\n")
    
    def test_constructor(self) -> Dict[str, Any]:
        """测试增强的Deep空间构造器"""
        print("=== 测试增强的Deep空间构造器 ===")
        
        results = {}
        
        # 测试不同聚类方法
        cluster_methods = ['kmeans', 'time_spatial', 'demand_balanced']
        
        for method in cluster_methods:
            print(f"\n测试聚类方法: {method}")
            
            try:
                # 创建构造器
                constructor = EnhancedDeepConstructor(self.instance, cluster_method=method)
                
                # 构造Deep空间
                start_time = time.time()
                deep_nodes = constructor.construct_deep_space(target_clusters=6)
                construction_time = time.time() - start_time
                
                # 分析Deep节点
                node_analysis = self._analyze_deep_nodes(deep_nodes)
                
                results[method] = {
                    'node_count': len(deep_nodes),
                    'construction_time': construction_time,
                    'node_analysis': node_analysis,
                    'success': True
                }
                
                print(f"  节点数量: {len(deep_nodes)}")
                print(f"  构造时间: {construction_time:.3f}秒")
                print(f"  平均节点大小: {node_analysis['avg_size']:.2f}")
                print(f"  平均需求: {node_analysis['avg_demand']:.2f}")
                
            except Exception as e:
                print(f"  错误: {str(e)}")
                results[method] = {
                    'success': False,
                    'error': str(e)
                }
        
        self.test_results['constructor'] = results
        return results
    
    def test_decoder(self) -> Dict[str, Any]:
        """测试Deep空间解码器"""
        print("=== 测试Deep空间解码器 ===")
        
        results = {}
        
        # 生成测试染色体
        deep_nodes = self.constructor.deep_nodes
        test_chromosomes = self._generate_test_chromosomes(deep_nodes)
        
        # 测试不同解码策略
        decode_strategies = ['nearest_neighbor', 'time_oriented', 'savings', 'hybrid']
        
        for strategy in decode_strategies:
            print(f"\n测试解码策略: {strategy}")
            
            strategy_results = []
            
            for i, chromosome in enumerate(test_chromosomes):
                try:
                    # 解码
                    start_time = time.time()
                    decoded_solution = self.decoder.decode_to_full_solution(
                        chromosome, strategy=strategy
                    )
                    decode_time = time.time() - start_time
                    
                    # 评估解
                    evaluation = self.decoder.evaluate_solution(decoded_solution)
                    
                    strategy_results.append({
                        'chromosome_id': i,
                        'decode_time': decode_time,
                        'total_distance': evaluation['total_distance'],
                        'route_count': evaluation['route_count'],
                        'feasible': evaluation['feasible'],
                        'success': True
                    })
                    
                    print(f"  染色体{i}: 距离={evaluation['total_distance']:.2f}, "
                          f"路径数={evaluation['route_count']}, "
                          f"可行={evaluation['feasible']}")
                    
                except Exception as e:
                    print(f"  染色体{i}解码失败: {str(e)}")
                    strategy_results.append({
                        'chromosome_id': i,
                        'success': False,
                        'error': str(e)
                    })
            
            # 计算策略统计
            successful_results = [r for r in strategy_results if r.get('success', False)]
            if successful_results:
                avg_distance = np.mean([r['total_distance'] for r in successful_results])
                avg_routes = np.mean([r['route_count'] for r in successful_results])
                avg_time = np.mean([r['decode_time'] for r in successful_results])
                feasible_rate = np.mean([r['feasible'] for r in successful_results])
                
                results[strategy] = {
                    'avg_distance': avg_distance,
                    'avg_routes': avg_routes,
                    'avg_decode_time': avg_time,
                    'feasible_rate': feasible_rate,
                    'success_rate': len(successful_results) / len(test_chromosomes),
                    'detailed_results': strategy_results
                }
                
                print(f"  平均距离: {avg_distance:.2f}")
                print(f"  平均路径数: {avg_routes:.2f}")
                print(f"  平均解码时间: {avg_time:.4f}秒")
                print(f"  可行性率: {feasible_rate:.2%}")
            else:
                results[strategy] = {
                    'success_rate': 0.0,
                    'detailed_results': strategy_results
                }
        
        self.test_results['decoder'] = results
        return results
    
    def test_evolution_engine(self) -> Dict[str, Any]:
        """测试Deep空间进化算法引擎"""
        print("=== 测试Deep空间进化算法引擎 ===")
        
        results = {}
        
        try:
            # 设置进化参数
            evolution_params = {
                'max_generations': 20,
                'population_size': 15,
                'elite_size': 3,
                'diversity_threshold': 0.1
            }
            
            print(f"进化参数: {evolution_params}")
            
            # 运行进化算法
            start_time = time.time()
            best_individual, evolution_history = self.evolution_engine.evolve(
                **evolution_params
            )
            evolution_time = time.time() - start_time
            
            # 解码最佳个体
            best_solution = self.decoder.decode_to_full_solution(
                best_individual.chromosome, strategy='hybrid'
            )
            best_evaluation = self.decoder.evaluate_solution(best_solution)
            
            # 分析进化历史
            fitness_history = [gen['best_fitness'] for gen in evolution_history]
            diversity_history = [gen['diversity'] for gen in evolution_history]
            
            results = {
                'success': True,
                'evolution_time': evolution_time,
                'generations': len(evolution_history),
                'best_fitness': best_individual.fitness,
                'best_distance': best_evaluation['total_distance'],
                'best_routes': best_evaluation['route_count'],
                'best_feasible': best_evaluation['feasible'],
                'fitness_improvement': fitness_history[0] - fitness_history[-1] if len(fitness_history) > 1 else 0,
                'final_diversity': diversity_history[-1] if diversity_history else 0,
                'convergence_generation': self._find_convergence_point(fitness_history),
                'evolution_history': evolution_history
            }
            
            print(f"进化时间: {evolution_time:.2f}秒")
            print(f"进化代数: {len(evolution_history)}")
            print(f"最佳适应度: {best_individual.fitness:.4f}")
            print(f"最佳距离: {best_evaluation['total_distance']:.2f}")
            print(f"最佳路径数: {best_evaluation['route_count']}")
            print(f"解可行性: {best_evaluation['feasible']}")
            print(f"适应度改进: {results['fitness_improvement']:.4f}")
            
        except Exception as e:
            print(f"进化算法测试失败: {str(e)}")
            results = {
                'success': False,
                'error': str(e)
            }
        
        self.test_results['evolution'] = results
        return results
    
    def test_encoder(self) -> Dict[str, Any]:
        """测试Deep空间状态编码器"""
        print("=== 测试Deep空间状态编码器 ===")
        
        results = {}
        
        try:
            # 生成测试染色体
            deep_nodes = self.constructor.deep_nodes
            test_chromosomes = self._generate_test_chromosomes(deep_nodes)
            
            # 拟合编码器
            print("拟合状态编码器...")
            start_time = time.time()
            self.encoder.fit_preprocessing(test_chromosomes)
            fit_time = time.time() - start_time
            
            # 测试编码
            encoding_times = []
            encoded_states = []
            
            for i, chromosome in enumerate(test_chromosomes):
                start_time = time.time()
                state_vector = self.encoder.encode_state(chromosome)
                encode_time = time.time() - start_time
                
                encoding_times.append(encode_time)
                encoded_states.append(state_vector)
                
                print(f"  染色体{i}: 编码时间={encode_time:.4f}秒, "
                      f"特征维度={len(state_vector)}")
            
            # 分析编码结果
            avg_encode_time = np.mean(encoding_times)
            feature_dim = len(encoded_states[0]) if encoded_states else 0
            
            # 计算特征统计
            if encoded_states:
                encoded_matrix = np.array(encoded_states)
                feature_means = np.mean(encoded_matrix, axis=0)
                feature_stds = np.std(encoded_matrix, axis=0)
                feature_ranges = np.max(encoded_matrix, axis=0) - np.min(encoded_matrix, axis=0)
                
                # 特征重要性
                importance = self.encoder.get_feature_importance()
                
                results = {
                    'success': True,
                    'fit_time': fit_time,
                    'avg_encode_time': avg_encode_time,
                    'feature_dimension': feature_dim,
                    'feature_statistics': {
                        'mean_range': np.mean(feature_ranges),
                        'std_range': np.std(feature_ranges),
                        'zero_variance_features': np.sum(feature_stds == 0),
                        'high_variance_features': np.sum(feature_stds > 1.0)
                    },
                    'feature_importance': importance,
                    'encoding_times': encoding_times
                }
                
                print(f"拟合时间: {fit_time:.3f}秒")
                print(f"平均编码时间: {avg_encode_time:.4f}秒")
                print(f"特征维度: {feature_dim}")
                print(f"零方差特征数: {results['feature_statistics']['zero_variance_features']}")
                print(f"高方差特征数: {results['feature_statistics']['high_variance_features']}")
            else:
                results = {
                    'success': False,
                    'error': '无法生成编码状态'
                }
                
        except Exception as e:
            print(f"状态编码器测试失败: {str(e)}")
            results = {
                'success': False,
                'error': str(e)
            }
        
        self.test_results['encoder'] = results
        return results
    
    def test_integration(self) -> Dict[str, Any]:
        """测试系统集成"""
        print("=== 测试系统集成 ===")
        
        results = {}
        
        try:
            # 完整的Deep空间求解流程
            print("执行完整的Deep空间求解流程...")
            
            # 1. 构造Deep空间
            start_time = time.time()
            deep_nodes = self.constructor.construct_deep_space(target_clusters=6)
            construction_time = time.time() - start_time
            
            # 2. 初始化编码器
            sample_chromosomes = self._generate_test_chromosomes(deep_nodes, count=10)
            self.encoder.fit_preprocessing(sample_chromosomes)
            
            # 3. 运行进化算法
            evolution_start = time.time()
            best_individual, evolution_history = self.evolution_engine.evolve(
                max_generations=15,
                population_size=12,
                elite_size=2
            )
            evolution_time = time.time() - evolution_start
            
            # 4. 解码最终解
            decode_start = time.time()
            final_solution = self.decoder.decode_to_full_solution(
                best_individual.chromosome, strategy='hybrid'
            )
            decode_time = time.time() - decode_start
            
            # 5. 评估最终解
            final_evaluation = self.decoder.evaluate_solution(final_solution)
            
            # 6. 编码最终状态
            encode_start = time.time()
            final_state = self.encoder.encode_state(best_individual.chromosome, final_solution)
            encode_time = time.time() - encode_start
            
            total_time = time.time() - start_time
            
            results = {
                'success': True,
                'total_time': total_time,
                'construction_time': construction_time,
                'evolution_time': evolution_time,
                'decode_time': decode_time,
                'encode_time': encode_time,
                'deep_nodes_count': len(deep_nodes),
                'evolution_generations': len(evolution_history),
                'final_fitness': best_individual.fitness,
                'final_distance': final_evaluation['total_distance'],
                'final_routes': final_evaluation['route_count'],
                'final_feasible': final_evaluation['feasible'],
                'state_dimension': len(final_state),
                'performance_metrics': {
                    'construction_efficiency': len(deep_nodes) / construction_time,
                    'evolution_efficiency': len(evolution_history) / evolution_time,
                    'decode_efficiency': final_evaluation['route_count'] / decode_time,
                    'overall_efficiency': final_evaluation['total_distance'] / total_time
                }
            }
            
            print(f"总时间: {total_time:.2f}秒")
            print(f"  构造时间: {construction_time:.3f}秒")
            print(f"  进化时间: {evolution_time:.2f}秒")
            print(f"  解码时间: {decode_time:.4f}秒")
            print(f"  编码时间: {encode_time:.4f}秒")
            print(f"Deep节点数: {len(deep_nodes)}")
            print(f"进化代数: {len(evolution_history)}")
            print(f"最终距离: {final_evaluation['total_distance']:.2f}")
            print(f"最终路径数: {final_evaluation['route_count']}")
            print(f"解可行性: {final_evaluation['feasible']}")
            
        except Exception as e:
            print(f"系统集成测试失败: {str(e)}")
            results = {
                'success': False,
                'error': str(e)
            }
        
        self.test_results['integration'] = results
        return results
    
    def _analyze_deep_nodes(self, deep_nodes: List) -> Dict[str, float]:
        """分析Deep节点特性"""
        if not deep_nodes:
            return {}
        
        sizes = [len(node.original_customers) for node in deep_nodes]
        demands = [node.total_demand for node in deep_nodes]
        service_times = [node.internal_service_time for node in deep_nodes]
        
        return {
            'avg_size': np.mean(sizes),
            'std_size': np.std(sizes),
            'avg_demand': np.mean(demands),
            'std_demand': np.std(demands),
            'avg_service_time': np.mean(service_times),
            'total_customers': sum(sizes)
        }
    
    def _generate_test_chromosomes(self, deep_nodes: List, count: int = 5) -> List[List[List[int]]]:
        """生成测试染色体"""
        chromosomes = []
        node_count = len(deep_nodes)
        
        for _ in range(count):
            # 随机生成染色体
            chromosome = []
            used_nodes = set()
            
            # 生成随机路径数
            route_count = np.random.randint(2, min(6, node_count + 1))
            
            for _ in range(route_count):
                # 生成随机路径
                available_nodes = [i for i in range(node_count) if i not in used_nodes]
                if not available_nodes:
                    break
                
                route_length = np.random.randint(1, min(4, len(available_nodes) + 1))
                route = np.random.choice(available_nodes, size=route_length, replace=False).tolist()
                
                chromosome.append(route)
                used_nodes.update(route)
            
            chromosomes.append(chromosome)
        
        return chromosomes
    
    def _find_convergence_point(self, fitness_history: List[float]) -> int:
        """找到收敛点"""
        if len(fitness_history) < 3:
            return len(fitness_history)
        
        # 寻找适应度改进停滞的点
        improvement_threshold = 0.001
        
        for i in range(2, len(fitness_history)):
            recent_improvement = abs(fitness_history[i-2] - fitness_history[i])
            if recent_improvement < improvement_threshold:
                return i
        
        return len(fitness_history)
    
    def generate_report(self) -> str:
        """生成测试报告"""
        report = []
        report.append("=" * 60)
        report.append("Deep空间独立系统测试报告")
        report.append("=" * 60)
        report.append(f"测试实例: {self.instance_file}")
        report.append(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 构造器测试结果
        if 'constructor' in self.test_results:
            report.append("1. 增强Deep空间构造器测试")
            report.append("-" * 40)
            constructor_results = self.test_results['constructor']
            
            for method, result in constructor_results.items():
                if result.get('success', False):
                    report.append(f"  {method}:")
                    report.append(f"    节点数量: {result['node_count']}")
                    report.append(f"    构造时间: {result['construction_time']:.3f}秒")
                    report.append(f"    平均节点大小: {result['node_analysis']['avg_size']:.2f}")
                else:
                    report.append(f"  {method}: 失败 - {result.get('error', '未知错误')}")
            report.append("")
        
        # 解码器测试结果
        if 'decoder' in self.test_results:
            report.append("2. Deep空间解码器测试")
            report.append("-" * 40)
            decoder_results = self.test_results['decoder']
            
            for strategy, result in decoder_results.items():
                if result.get('success_rate', 0) > 0:
                    report.append(f"  {strategy}:")
                    report.append(f"    成功率: {result['success_rate']:.2%}")
                    report.append(f"    平均距离: {result['avg_distance']:.2f}")
                    report.append(f"    平均路径数: {result['avg_routes']:.2f}")
                    report.append(f"    可行性率: {result['feasible_rate']:.2%}")
                else:
                    report.append(f"  {strategy}: 失败")
            report.append("")
        
        # 进化算法测试结果
        if 'evolution' in self.test_results:
            report.append("3. Deep空间进化算法测试")
            report.append("-" * 40)
            evolution_results = self.test_results['evolution']
            
            if evolution_results.get('success', False):
                report.append(f"  进化时间: {evolution_results['evolution_time']:.2f}秒")
                report.append(f"  进化代数: {evolution_results['generations']}")
                report.append(f"  最佳适应度: {evolution_results['best_fitness']:.4f}")
                report.append(f"  最佳距离: {evolution_results['best_distance']:.2f}")
                report.append(f"  适应度改进: {evolution_results['fitness_improvement']:.4f}")
            else:
                report.append(f"  测试失败: {evolution_results.get('error', '未知错误')}")
            report.append("")
        
        # 编码器测试结果
        if 'encoder' in self.test_results:
            report.append("4. Deep空间状态编码器测试")
            report.append("-" * 40)
            encoder_results = self.test_results['encoder']
            
            if encoder_results.get('success', False):
                report.append(f"  拟合时间: {encoder_results['fit_time']:.3f}秒")
                report.append(f"  平均编码时间: {encoder_results['avg_encode_time']:.4f}秒")
                report.append(f"  特征维度: {encoder_results['feature_dimension']}")
                stats = encoder_results['feature_statistics']
                report.append(f"  零方差特征: {stats['zero_variance_features']}")
                report.append(f"  高方差特征: {stats['high_variance_features']}")
            else:
                report.append(f"  测试失败: {encoder_results.get('error', '未知错误')}")
            report.append("")
        
        # 集成测试结果
        if 'integration' in self.test_results:
            report.append("5. 系统集成测试")
            report.append("-" * 40)
            integration_results = self.test_results['integration']
            
            if integration_results.get('success', False):
                report.append(f"  总执行时间: {integration_results['total_time']:.2f}秒")
                report.append(f"  Deep节点数: {integration_results['deep_nodes_count']}")
                report.append(f"  进化代数: {integration_results['evolution_generations']}")
                report.append(f"  最终距离: {integration_results['final_distance']:.2f}")
                report.append(f"  最终路径数: {integration_results['final_routes']}")
                report.append(f"  解可行性: {integration_results['final_feasible']}")
                
                metrics = integration_results['performance_metrics']
                report.append("  性能指标:")
                report.append(f"    构造效率: {metrics['construction_efficiency']:.2f} 节点/秒")
                report.append(f"    进化效率: {metrics['evolution_efficiency']:.2f} 代/秒")
                report.append(f"    解码效率: {metrics['decode_efficiency']:.2f} 路径/秒")
            else:
                report.append(f"  测试失败: {integration_results.get('error', '未知错误')}")
            report.append("")
        
        # 总结
        report.append("6. 测试总结")
        report.append("-" * 40)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() 
                             if isinstance(result, dict) and result.get('success', False))
        
        report.append(f"  总测试模块: {total_tests}")
        report.append(f"  成功模块: {successful_tests}")
        report.append(f"  成功率: {successful_tests/total_tests:.2%}" if total_tests > 0 else "  成功率: 0%")
        
        if successful_tests == total_tests:
            report.append("  ✅ Deep空间独立系统运行正常！")
        else:
            report.append("  ⚠️  部分模块存在问题，需要进一步调试。")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_results(self, output_file: str = "deep_system_test_results.json"):
        """保存测试结果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"测试结果已保存到: {output_file}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("开始Deep空间独立系统全面测试...\n")
        
        # 设置系统
        self.setup_system()
        
        # 运行各项测试
        self.test_constructor()
        print()
        
        self.test_decoder()
        print()
        
        self.test_evolution_engine()
        print()
        
        self.test_encoder()
        print()
        
        self.test_integration()
        print()
        
        # 生成报告
        report = self.generate_report()
        print(report)
        
        # 保存结果
        self.save_results()
        
        return self.test_results


def main():
    """主函数"""
    # 检查实例文件
    instance_files = [
        "data/RC1_2_1.txt",
        "data/C1_2_1.txt",
        "data/R1_2_1.txt"
    ]
    
    # 寻找可用的实例文件
    available_file = None
    for file_path in instance_files:
        if os.path.exists(file_path):
            available_file = file_path
            break
    
    if not available_file:
        print("错误: 找不到VRPTW实例文件")
        print("请确保以下文件之一存在:")
        for file_path in instance_files:
            print(f"  - {file_path}")
        return
    
    # 创建测试器并运行测试
    tester = DeepSystemTester(available_file)
    
    try:
        results = tester.run_all_tests()
        print("\n✅ Deep空间独立系统测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()