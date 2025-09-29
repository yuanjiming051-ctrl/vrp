#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Deep空间聚合的时间窗逻辑
"""

import numpy as np
from vrptw_instance import VRPTWInstance
from deep_constructor import DeepVRPTWInstance

def test_deep_aggregation():
    """测试Deep空间聚合的时间窗计算"""
    print("=== 测试Deep空间聚合时间窗逻辑 ===")
    
    # 加载测试实例
    instance_file = "solomon_100/c101.txt"
    try:
        inst = VRPTWInstance(instance_file)
        print(f"成功加载实例: {instance_file}")
        print(f"客户数量: {len(inst.ordinary_customers)}")
        print(f"车辆容量: {inst.vehicle_info['capacity']}")
        
        # 创建Deep空间实例
        deep_inst = DeepVRPTWInstance(inst)
        print(f"\nDeep空间客户数量: {len(deep_inst.ordinary_customers)}")
        
        # 显示前几个聚合客户的详细信息
        print("\n=== 聚合客户详细信息 ===")
        for i, customer in enumerate(deep_inst.ordinary_customers[:5]):
            print(f"\n聚合客户 {i+1}:")
            print(f"  位置: ({customer['xcoord']:.2f}, {customer['ycoord']:.2f})")
            print(f"  需求: {customer['demand']}")
            print(f"  服务时间: {customer['service_time']:.2f}")
            print(f"  时间窗: [{customer['ready_time']:.0f}, {customer['due_date']:.0f}]")
            
            # 检查时间窗合理性
            if customer['ready_time'] >= customer['due_date']:
                print(f"  ⚠️  警告: 时间窗不合理! ready_time >= due_date")
            else:
                window_size = customer['due_date'] - customer['ready_time']
                print(f"  ✓ 时间窗大小: {window_size:.0f}")
        
        # 检查原始路径信息
        print(f"\n=== 原始聚合路径信息 ===")
        if hasattr(deep_inst, 'VC_new'):
            for i, route in enumerate(deep_inst.VC_new[:3]):
                print(f"\n路径 {i+1}: {route}")
                if len(route) > 1:
                    # 计算原始路径的时间窗可行性
                    total_time = 0
                    current_time = 0
                    
                    for j, customer_idx in enumerate(route):
                        customer = inst.ordinary_customers[customer_idx]
                        if j == 0:
                            # 从仓库到第一个客户
                            travel_time = deep_inst.D[0, customer_idx + 1]
                            arrival_time = max(travel_time, customer['ready_time'])
                        else:
                            # 从前一个客户到当前客户
                            prev_idx = route[j-1]
                            travel_time = deep_inst.D[prev_idx + 1, customer_idx + 1]
                            arrival_time = max(current_time + travel_time, customer['ready_time'])
                        
                        departure_time = arrival_time + customer['service_time']
                        current_time = departure_time
                        
                        print(f"  客户{customer_idx}: 到达{arrival_time:.1f}, 离开{departure_time:.1f}, 时间窗[{customer['ready_time']}, {customer['due_date']}]")
                        
                        if arrival_time > customer['due_date']:
                            print(f"    ⚠️  违反时间窗约束!")
                    
                    # 返回仓库
                    last_customer_idx = route[-1]
                    return_time = deep_inst.D[last_customer_idx + 1, 0]
                    total_time = current_time + return_time
                    print(f"  总时间: {total_time:.1f}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_deep_aggregation()