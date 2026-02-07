import math


def f(x: float) -> float:
    """f(x) = e^x - 5x"""
    return math.exp(x) - 5.0 * x


def fp(x: float) -> float:
    """f'(x) = e^x - 5"""
    return math.exp(x) - 5.0


def solve_on_interval(a: float = 1.0, b: float = 2.0):
    # 1) 内部驻点
    x_star = math.log(5.0)

    # 2) 候选点集合（端点必算）
    candidates = [a, b]
    if a <= x_star <= b:
        candidates.append(x_star)

    # 3) 计算并选最小
    best_x = None
    best_val = None
    for x in candidates:
        val = f(x)
        if best_val is None or val < best_val:
            best_val = val
            best_x = x

    return best_x, best_val, candidates


if __name__ == "__main__":
    left, right = 1.0, 2.0

    x_min, f_min, cand = solve_on_interval(left, right)

    print("Interval:", [left, right])
    print(f"Minimizer x* = {x_min:.12f}")
    print(f"Minimum  f(x*) = {f_min:.12f}")
