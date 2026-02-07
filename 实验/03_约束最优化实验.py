import numpy as np
from scipy.optimize import minimize

# NOTE: 题面边界处 x4 重复、x5 缺失；按“第二个 x4 实为 x5”处理：0 <= x5 <= 6
# 若老师有勘误，以原题为准，只需改 bounds。

# ---- 可改参数（最重要）----
x0 = np.array([2.0, 2.0, 3.0, 5.0, 5.0, 10.0])  # 初值（建议取可行点）
ftol = 1e-9
maxiter = 500
CHECK = False  # 想看约束余量/边界是否顶到头，就改 True

bounds = [
    (0.0, None),   # x1 >= 0
    (0.0, None),   # x2 >= 0
    (1.0, 5.0),    # x3
    (1.0, 5.0),    # x4
    (0.0, 6.0),    # x5（修复）
    (0.0, 10.0),   # x6
]


def obj(x):
    x1, x2, x3, x4, x5, x6 = x
    return (
        -25.0 * (x1 - 2.0) ** 2
        - (x2 - 2.0) ** 2
        - (x3 - 1.0) ** 2
        - (x4 - 4.0) ** 2
        - (x5 - 1.0) ** 2
        - (x6 - 4.0) ** 2
    )


# SLSQP: ineq 约束形式必须是 fun(x) >= 0
cons = [
    {"type": "ineq", "fun": lambda x: (x[2] - 3.0) ** 2 + x[3] - 4.0},      # (x3-3)^2 + x4 >= 4
    {"type": "ineq", "fun": lambda x: (x[4] - 5.0) ** 2 + x[5] - 4.0},      # (x5-5)^2 + x6 >= 4
    {"type": "ineq", "fun": lambda x: 2.0 - x[0] + 3.0 * x[1]},             # x1 - 3x2 <= 2
    {"type": "ineq", "fun": lambda x: 2.0 + x[0] - x[1]},                   # -x1 + x2 <= 2
    {"type": "ineq", "fun": lambda x: x[0] + x[1] - 2.0},                   # x1 + x2 >= 2
    {"type": "ineq", "fun": lambda x: 6.0 - (x[0] + x[1])},                 # x1 + x2 <= 6
]


def check(x):
    # 约束余量（>=0 表示满足）
    r = np.array([
        (x[2] - 3.0) ** 2 + x[3] - 4.0,
        (x[4] - 5.0) ** 2 + x[5] - 4.0,
        2.0 - x[0] + 3.0 * x[1],
        2.0 + x[0] - x[1],
        x[0] + x[1] - 2.0,
        6.0 - (x[0] + x[1]),
    ])
    return r


if __name__ == "__main__":
    res = minimize(
        obj,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"ftol": ftol, "maxiter": maxiter, "disp": False},
    )

    print("success =", res.success)
    print("x* =", np.array2string(res.x, precision=6, suppress_small=True))
    print("z* =", float(res.fun))

    if CHECK:
        r = check(res.x)
        print("min constraint residual =", float(r.min()))  # <0 就违反约束
        # 看哪些变量顶到边界（接近边界就会被标出来）
        eps = 1e-6
        for i, (lo, up) in enumerate(bounds):
            xi = res.x[i]
            hit_lo = (lo is not None) and (xi <= lo + eps)
            hit_up = (up is not None) and (xi >= up - eps)
            if hit_lo or hit_up:
                print(f"x{i+1} hits bound: x={xi:.6f}, bound=({lo},{up})")
