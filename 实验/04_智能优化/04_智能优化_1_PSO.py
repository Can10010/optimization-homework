import numpy as np

LB, UB = -4.0, 10.0          # 边界：-4 <= x1,x2 <= 10
DIM = 2                      # 变量维度：x=[x1,x2]

N = 40                       # 粒子数
MAX_IT = 200                 # 迭代次数

w = 0.72                     # 惯性权重
c1 = 1.49                    # 个体学习因子
c2 = 1.49                    # 社会学习因子

SEED = 0                     # 随机种子（想换就换）


def f(x):
    x1, x2 = x[..., 0], x[..., 1]
    return 4 * x1**2 + 5 * x2**2 - 3 * x1 - 2 * x2 + 7


def clamp(X, lb=LB, ub=UB):
    return np.minimum(np.maximum(X, lb), ub)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)

    # 初始化位置、速度
    X = rng.uniform(LB, UB, size=(N, DIM))
    V = rng.uniform(-(UB - LB), (UB - LB), size=(N, DIM)) * 0.1

    # 初始化个体最优、全局最优
    pbest = X.copy()
    pbest_val = f(pbest)

    g_idx = np.argmin(pbest_val)
    gbest = pbest[g_idx].copy()
    gbest_val = float(pbest_val[g_idx])

    for _ in range(MAX_IT):
        r1 = rng.random((N, DIM))
        r2 = rng.random((N, DIM))

        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = X + V

        # 截断法处理边界
        X = clamp(X, LB, UB)

        # 更新 pbest / gbest
        vals = f(X)
        better = vals < pbest_val
        pbest[better] = X[better]
        pbest_val[better] = vals[better]

        g_idx = np.argmin(pbest_val)
        if pbest_val[g_idx] < gbest_val:
            gbest_val = float(pbest_val[g_idx])
            gbest = pbest[g_idx].copy()

    print("PSO best x =", gbest)
    print("PSO best z =", gbest_val)
