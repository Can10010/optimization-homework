import numpy as np
from scipy.optimize import linprog

c = [-2, -1, 1]  # 转换为 min -2x1 - x2 + x3

# 不等式约束矩阵（<=）
A = [[1, 1, 2],   
     [1, 4, -1]]  
b = [6, 4]

x0_bounds = (0, None)
x1_bounds = (0, None)
x2_bounds = (0, None)

res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds, x2_bounds], method='highs')

print("最优解:")
print(f"x1 = {res.x[0]:.6f}")
print(f"x2 = {res.x[1]:.6f}")
print(f"x3 = {res.x[2]:.6f}")
print(f"最大值 = {-res.fun:.6f}")  # 因为求的是 min -f，所以最大值是 -fun


