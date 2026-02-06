import numpy as np
from scipy.optimize import linprog


# ====== 随机生成数据：m=10种资源，n=4种产品，且满足“合同下界可行” ======
m, n = 10, 4
rng = np.random.default_rng(0)

# 资源总量 b_i
b = rng.integers(200, 400, size=m).astype(float)

# 消耗系数 a_ij（资源 i 用于产品 j 的消耗）
A = rng.integers(1, 20, size=(m, n)).astype(float)

# 合同下界 d_j（最低产量）
d = rng.integers(5, 30, size=n).astype(float)

# 单位收益 c_j
c = rng.integers(10, 80, size=n).astype(float)

# 调整 b 使得至少能满足合同下界：A @ d <= b
need = A @ d
b = np.maximum(b, need + rng.integers(0, 50, size=m))

# ====== 线性规划：max c^T x  ->  min (-c)^T x ======
# 资源约束：A x <= b
A_ub = A
b_ub = b

# 变量下界：x_j >= d_j
bounds = [(float(d[j]), None) for j in range(n)]

res = linprog(
    c=-c,               # 最大化转最小化
    A_ub=A_ub,
    b_ub=b_ub,
    bounds=bounds,
    method="highs"
)

x = res.x
profit = -res.fun

print("max revenue =", profit)
print("x =", x)
