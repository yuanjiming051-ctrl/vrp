
# route_optimizer.py

def route_cost(route, dist_mat):
    """
    计算单条子路由的总距离：depot → route → depot
    :param route: List[int]，客户索引列表（0-based，对应 dist_mat 中的行/列要 +1）
    :param dist_mat: 2D array，大小 (N+1)×(N+1)，dist_mat[0] 是 depot
    :return: float，总距离
    """
    cost = 0.0
    prev = 0  # 从 depot（索引 0）出发
    for cust in route:
        # 客户在 dist_mat 中索引为 cust+1
        cost += dist_mat[prev][cust + 1]
        prev = cust + 1
    # 返回 depot
    cost += dist_mat[prev][0]
    return cost


def two_opt_route(route, dist_mat):
    """
    对单条子路由做 2-opt 改进
    :param route: List[int]，客户索引列表
    :param dist_mat: 2D array，(N+1)x(N+1) 距离矩阵
    :return: List[int]，优化后的客户索引列表
    """
    best = route[:]
    best_cost = route_cost(best, dist_mat)
    improved = True

    # 重复直到没有改进为止
    while improved:
        improved = False
        n = len(best)
        # i 从 1 开始保留 depot→第一个客户不变，j 最多到 n-1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # 生成新解：翻转区间 [i, j]
                new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                new_cost = route_cost(new_route, dist_mat)
                if new_cost < best_cost:
                    best = new_route
                    best_cost = new_cost
                    improved = True
                    # 一旦有改进，跳出内层循环，重新从头开始
                    break
            if improved:
                break

    return best