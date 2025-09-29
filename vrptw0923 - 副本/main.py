# File: main.py

import matplotlib.pyplot as plt

from vrptw_instance import VRPTWInstance
from decoder import VRPTWDecoder
from dqn_alns_solver import DQNALNSSolver


def main():
    # 请根据实际路径修改
    file_path = 'data/RC1_2_1.txt'

    inst = VRPTWInstance(file_path)
    if not inst.vehicle_info or not inst.ordinary_customers:
        print("数据读取失败，程序终止")
        return

    # 创建DQNALNSSolver（优化参数以提高求解质量）
    solver = DQNALNSSolver(
        instance=inst,
        pop_size=50,  # 种群大小
        delta=5,
        deep_iters=150,  # Deep空间迭代次数（
        deep_search_multiplier=1,  # 增加降维空间搜索次数倍数
        iters=20,
        full_iters=100,  # Full空间迭代次数
        eps_start=1.0,
        eps_end=0.1,  # 进一步降低最终探索率，增强利用
        eps_decay=0.9995,  # 更慢的衰减，保持更长时间的探索
        enable_local_search=True,
        local_search_type='adaptive',
        local_search_frequency=1,  # 每轮都进行局部搜索（从2改为1）
        # 多样化策略参数
        enable_restart=True,  # 启用重启机制
        restart_threshold=100,  # 100轮无改进后考虑重启
        diversity_threshold=0.1,  # 种群多样性阈值
        elite_ratio=0.2,  # 重启时保留20%的精英个体
        verbose=True
    )

    # 现在 solve 返回 5 个值：最优序列、最优距离、历史距离、10分钟快照、30分钟快照
    best_seq, best_cost, cost_history, ten_min_snapshot, thirty_min_snapshot = solver.solve()

    # 解码最终解以获取车辆数等信息
    decoder = VRPTWDecoder(inst)
    decoded = decoder.decode_solution(best_seq)
    veh_count = decoded.get('vehicle_count', None)

    print("\n=== 最终结果 ===")
    print("最优路径序列：", best_seq)
    if veh_count is not None:
        print(f"最优车辆数： {veh_count}")
    print(f"最优总距离： {best_cost:.2f}")

    # 1) 成本（距离）下降曲线（y 轴依旧为距离，目标已在求解器内部以字典序实现）
    plt.figure(figsize=(8, 5))
    iters = list(range(1, len(cost_history) + 1))
    plt.plot(iters, cost_history, '-o', linewidth=2, markersize=4)
    plt.xlabel('Iteration')
    plt.ylabel('Best Distance')
    plt.title('Distance Convergence (Vehicles-first Objective)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # 2) 最终解可视化
    routes = decoded['routes']
    title_suffix = f"Vehicles = {veh_count}, Distance = {best_cost:.2f}" if veh_count is not None else f"Distance = {best_cost:.2f}"
    decoder.visualize_routes(
        routes, inst,
        title=f'Final VRPTW Routes ({title_suffix})'
    )

    # 3) 可选：打印 10/30 分钟快照
    if ten_min_snapshot is not None:
        print("\n[10分钟快照]")
        print("  最优距离：", f"{ten_min_snapshot['best_cost']:.2f}")
    if thirty_min_snapshot is not None:
        print("\n[30分钟快照]")
        print("  最优距离：", f"{thirty_min_snapshot['best_cost']:.2f}")


if __name__ == "__main__":
    main()
