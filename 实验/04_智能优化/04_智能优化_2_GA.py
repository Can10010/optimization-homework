import numpy as np

# =========================
# 可改参数（老师最爱改的）
# =========================
LB, UB = -4.0, 10.0
DIM = 2

POP = 60                     # 种群规模
GEN = 200                    # 代数

TOUR = 3                     # 锦标赛规模（选择压力）
PC = 0.9                     # 交叉概率
PM = 0.2                     # 变异概率（按个体）
SIGMA = 0.4                  # 高斯变异强度（越大跳得越狠）

ELITE = 1                    # 精英保留个数
SEED = 0


def f(x):
    x1, x2 = x[..., 0], x[..., 1]
    return 4 * x1**2 + 5 * x2**2 - 3 * x1 - 2 * x2 + 7


def clamp(X, lb=LB, ub=UB):
    return np.minimum(np.maximum(X, lb), ub)


def tournament_select(rng, pop, fit, k=TOUR):
    idx = rng.integers(0, len(pop), size=k)
    best = idx[np.argmin(fit[idx])]   # 这里是最小化
    return pop[best].copy()


def blend_crossover(rng, p1, p2, alpha=0.5):
    # 实数编码交叉：在父母附近做线性混合
    lam = rng.uniform(-alpha, 1 + alpha, size=p1.shape)
    c1 = lam * p1 + (1 - lam) * p2
    c2 = lam * p2 + (1 - lam) * p1
    return c1, c2


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)

    # 初始化种群
    pop = rng.uniform(LB, UB, size=(POP, DIM))
    pop = clamp(pop, LB, UB)

    for _ in range(GEN):
        fit = f(pop)

        # 精英保留
        elite_idx = np.argsort(fit)[:ELITE]
        new_pop = [pop[i].copy() for i in elite_idx]

        # 产生下一代
        while len(new_pop) < POP:
            p1 = tournament_select(rng, pop, fit)
            p2 = tournament_select(rng, pop, fit)

            if rng.random() < PC:
                c1, c2 = blend_crossover(rng, p1, p2, alpha=0.5)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # 变异（对每个子代独立）
            if rng.random() < PM:
                c1 = c1 + rng.normal(0.0, SIGMA, size=DIM)
            if rng.random() < PM:
                c2 = c2 + rng.normal(0.0, SIGMA, size=DIM)

            # 截断法边界处理
            c1 = clamp(c1, LB, UB)
            c2 = clamp(c2, LB, UB)

            new_pop.append(c1)
            if len(new_pop) < POP:
                new_pop.append(c2)

        pop = np.array(new_pop)

    fit = f(pop)
    best_idx = int(np.argmin(fit))
    best_x = pop[best_idx]
    best_z = float(fit[best_idx])

    print("GA best x =", best_x)
    print("GA best z =", best_z)

    # 参考真解（用来对照，不想要就删）
    # x* = (3/8, 1/5) = (0.375, 0.2)
