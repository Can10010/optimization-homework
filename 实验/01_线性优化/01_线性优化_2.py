import numpy as np
from scipy.optimize import linprog


m, n = 10, 4
rng = np.random.default_rng(0)

b = rng.integers(200, 400, size=m).astype(float)
A = rng.integers(1, 20, size=(m, n)).astype(float)
d = rng.integers(5, 30, size=n).astype(float)
c = rng.integers(10, 80, size=n).astype(float)

# 调整 b 使得至少能满足合同下界：A @ d <= b
need = A @ d
b = np.maximum(b, need + rng.integers(0, 50, size=m))

# 资源约束：A x <= b
A_ub = A
b_ub = b

# 变量下界：x_j >= d_j
bounds = [(float(d[j]), None) for j in range(n)]

res = linprog(
    c=-c,               
    A_ub=A_ub,
    b_ub=b_ub,
    bounds=bounds,
    method="highs"
)

x = res.x
profit = -res.fun

print("max revenue =", profit)
print("x =", x)
