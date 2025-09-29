#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断脚本：测试Full空间局部搜索和Deep空间搜索的修复效果
"""

import sys
import time
from dqn_alns_solver import DQNALNSSolver
from vrptw_instance import VRPTWInstance

def test_local_search_execution():
    """测试Full空间局部搜索是否正确执行"""
    print("=" * 60)
    print("测试1: Full空间局部搜索执行情况")
    print("=" * 60)
    
    # 使用小规模测试，减少迭代次数以便观察
    solver = DQNALNSSolver(
        instance_file="data/RC1_2_1.txt",
        pop_size=3,  # 小种群
        delta=3,
        deep_iters=10,  # 少量Deep迭代
        full_iters=5,   # 少量Full迭代，便于观察局部搜索
        local_search_frequency=1,  # 每次都执行局部搜索
        enable_local_search=True,
        local_search_type="adaptive",
        verbose=True  # 开启详细输出
    )
    
    print(f"配置: pop_size={solver.pop_size}, full_iters={solver.full_iters}")
    print(f"局部搜索: enable={solver.enable_local_search}, frequency={solver.local_search_frequency}")
    print(f"预期: 每个Full迭代都应该执行局部搜索")
    print()
    
    # 运行一轮测试
    start_time = time.time()
    best_solution, best_cost = solver.solve()
    end_time = time.time()
    
    print(f"\n测试完成，用时: {end_time - start_time:.2f}秒")
    print(f"最佳成本: {best_cost:.2f}")
    return best_cost

def test_deep_search_improvement():
    """测试Deep空间搜索的改进效果"""
    print("\n" + "=" * 60)
    print("测试2: Deep空间搜索改进效果")
    print("=" * 60)
    
    # 对比修复前后的参数设置
    print("修复前的参数（模拟）:")
    print("  - 破坏操作移除数量: len(chrom)//20, len(chrom)//15")
    print("  - Deep迭代次数: 50")
    print("  - 局部搜索: 简单2-opt")
    
    print("\n修复后的参数:")
    print("  - 破坏操作移除数量: len(chrom)//8, len(chrom)//6, len(chrom)//10")
    print("  - Deep迭代次数: 150")
    print("  - 局部搜索: 增强2-opt + relocate操作")
    
    solver = DQNALNSSolver(
        instance_file="data/RC1_2_1.txt",
        pop_size=3,
        delta=3,
        deep_iters=20,  # 适中的迭代次数用于测试
        full_iters=10,
        verbose=True
    )
    
    print(f"\n当前Deep空间破坏操作数量:")
    # 创建一个测试染色体来检查移除数量
    test_chrom = list(range(10))  # 假设10个Deep节点
    for i, op in enumerate(solver.destroy_ops_deep):
        # 模拟调用破坏操作（注意：这里只是为了展示参数）
        if i == 0:
            remove_count = max(2, len(test_chrom) // 8)
            print(f"  - 操作{i+1}: 移除 {remove_count} 个节点 (len//8)")
        elif i == 1:
            remove_count = max(2, len(test_chrom) // 6)
            print(f"  - 操作{i+1}: 移除 {remove_count} 个节点 (len//6)")
        elif i == 2:
            remove_count = max(2, len(test_chrom) // 10)
            print(f"  - 操作{i+1}: 移除 {remove_count} 个节点 (len//10)")
    
    start_time = time.time()
    best_solution, best_cost = solver.solve()
    end_time = time.time()
    
    print(f"\n测试完成，用时: {end_time - start_time:.2f}秒")
    print(f"最佳成本: {best_cost:.2f}")
    return best_cost

def test_parameter_sensitivity():
    """测试参数敏感性"""
    print("\n" + "=" * 60)
    print("测试3: 参数敏感性分析")
    print("=" * 60)
    
    configs = [
        {"name": "基础配置", "deep_iters": 50, "full_iters": 20},
        {"name": "增强Deep", "deep_iters": 150, "full_iters": 20},
        {"name": "增强Full", "deep_iters": 50, "full_iters": 50},
        {"name": "全面增强", "deep_iters": 150, "full_iters": 50}
    ]
    
    results = []
    
    for config in configs:
        print(f"\n测试配置: {config['name']}")
        print(f"  deep_iters={config['deep_iters']}, full_iters={config['full_iters']}")
        
        solver = DQNALNSSolver(
            instance_file="data/RC1_2_1.txt",
            pop_size=3,
            delta=3,
            deep_iters=config['deep_iters'],
            full_iters=config['full_iters'],
            verbose=False  # 关闭详细输出以减少噪音
        )
        
        start_time = time.time()
        best_solution, best_cost = solver.solve()
        end_time = time.time()
        
        results.append({
            "config": config['name'],
            "cost": best_cost,
            "time": end_time - start_time
        })
        
        print(f"  结果: 成本={best_cost:.2f}, 时间={end_time - start_time:.2f}秒")
    
    print(f"\n参数敏感性分析结果:")
    print("-" * 50)
    for result in results:
        print(f"{result['config']:12s}: 成本={result['cost']:8.2f}, 时间={result['time']:6.2f}秒")
    
    return results

def main():
    """主测试函数"""
    print("VRPTW算法诊断测试")
    print("测试修复后的Full空间局部搜索和Deep空间搜索效果")
    print("=" * 80)
    
    try:
        # 测试1: 局部搜索执行
        cost1 = test_local_search_execution()
        
        # 测试2: Deep空间改进
        cost2 = test_deep_search_improvement()
        
        # 测试3: 参数敏感性
        results = test_parameter_sensitivity()
        
        print("\n" + "=" * 80)
        print("诊断测试总结")
        print("=" * 80)
        print(f"测试1 (局部搜索): 最佳成本 = {cost1:.2f}")
        print(f"测试2 (Deep改进):  最佳成本 = {cost2:.2f}")
        print(f"测试3 (参数敏感性): 已完成多配置对比")
        
        print("\n修复效果评估:")
        print("1. Full空间局部搜索现在应该在每个Full迭代中正确执行")
        print("2. Deep空间搜索强度已增强（更多移除节点、更多迭代、更好的局部搜索）")
        print("3. 可以通过上述输出观察到具体的执行情况")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()