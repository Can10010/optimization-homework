import numpy as np
from scipy.optimize import linprog


def generate_balanced_ab(m, k, seed=None):
    rng = np.random.default_rng(seed)

    a = rng.integers(20, 80, size=m)
    total = int(a.sum())

    b = rng.integers(10, 60, size=k)
    b[-1] += total - int(b.sum())

    if b[-1] <= 0:
        return generate_balanced_ab(m, k, seed=rng.integers(1e9))

    return a.astype(float), b.astype(float)


def solve_transport(a, b, c):
    m, k = c.shape
    c_vec = c.reshape(-1)

    A_eq = np.zeros((m + k, m * k))

    for i in range(m):          # 供给
        A_eq[i, i * k:(i + 1) * k] = 1

    for j in range(k):          # 需求
        A_eq[m + j, j::k] = 1

    b_eq = np.concatenate([a, b])
    bounds = [(0, None)] * (m * k)

    res = linprog(c_vec, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")

    return res.x.reshape(m, k), res.fun


# ====== 题目给定规模，直接跑 ======
m, k = 10, 15
a, b = generate_balanced_ab(m, k, seed=0)
c = np.random.randint(1, 100, size=(m, k))

X, cost = solve_transport(a, b, c)

print("min cost =", cost)
print("X shape =", X.shape)
print(X)
