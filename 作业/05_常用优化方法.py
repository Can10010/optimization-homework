import numpy as np

Q = np.array([[4.0, 0.0],
              [0.0, 2.0]])
b = np.array([3.0, 0.0])
c = 2.0

def f(x):
    return 0.5 * x @ Q @ x - b @ x + c

def grad(x):
    return Q @ x - b

def hess(x):
    return Q


# ====== 打印辅助 ======
def report(name, hist):
    k, x, fx, ng = hist[-1]
    print(f"\n[{name}] iters={len(hist)-1}")
    print(f"  x = {x}")
    print(f"  f(x) = {fx}")
    print(f"  ||grad|| = {ng}")


# 1) 最速下降法（Steepest Descent）——对二次函数用“精确线搜索”
#    alpha_k = (g^T g) / (g^T Q g) , 方向 p=-g
def steepest_descent(x0, tol=1e-10, maxit=200):
    x = x0.copy()
    hist = []
    for k in range(maxit):
        g = grad(x)
        ng = np.linalg.norm(g)
        hist.append((k, x.copy(), f(x), ng))
        if ng < tol:
            break
        alpha = (g @ g) / (g @ (Q @ g))
        x = x - alpha * g
    return x, hist


# 2) 梯度下降法（Gradient Descent）——用固定步长（你也可以改成回溯线搜索）
#    注意：步长 alpha 要 < 2/L，这里 L=max(eig(Q))=4，所以 alpha<0.5
def gradient_descent(x0, alpha=0.4, tol=1e-10, maxit=5000):
    x = x0.copy()
    hist = []
    for k in range(maxit):
        g = grad(x)
        ng = np.linalg.norm(g)
        hist.append((k, x.copy(), f(x), ng))
        if ng < tol:
            break
        x = x - alpha * g
    return x, hist


# 3) 牛顿法（Newton）
#    p = -H^{-1} g。因为本题 H=Q 常数，所以理论上一步就到最优（数值上几乎一步）
def newton(x0, tol=1e-10, maxit=50):
    x = x0.copy()
    hist = []
    for k in range(maxit):
        g = grad(x)
        ng = np.linalg.norm(g)
        hist.append((k, x.copy(), f(x), ng))
        if ng < tol:
            break
        p = np.linalg.solve(Q, -g)
        x = x + p
    return x, hist


# 4) 共轭梯度法（Conjugate Gradient, CG）——专门解 Qx=b（等价最小化二次函数）
#    二维 SPD 情况下，最多 2 步（精确算术下）就收敛
def conjugate_gradient(x0, tol=1e-10, maxit=20):
    x = x0.copy()
    hist = []

    r = b - Q @ x          # r = -grad
    p = r.copy()
    rs = r @ r

    for k in range(maxit):
        g = grad(x)
        ng = np.linalg.norm(g)
        hist.append((k, x.copy(), f(x), ng))
        if np.sqrt(rs) < tol:
            break
        Ap = Q @ p
        alpha = rs / (p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        rs_new = r_new @ r_new
        beta = rs_new / rs
        p = r_new + beta * p
        r, rs = r_new, rs_new

    return x, hist


# 5) DFP（拟牛顿）——更新“逆 Hessian 近似” Hk
#    对二次函数 + 精确线搜索，它也会非常快（维度n内结束）
def dfp(x0, tol=1e-10, maxit=50):
    x = x0.copy()
    n = len(x)
    H = np.eye(n)
    hist = []

    for k in range(maxit):
        g = grad(x)
        ng = np.linalg.norm(g)
        hist.append((k, x.copy(), f(x), ng))
        if ng < tol:
            break

        p = -(H @ g)

        # 二次函数下的精确线搜索
        alpha = -(g @ p) / (p @ (Q @ p))
        x_new = x + alpha * p

        s = x_new - x
        y = grad(x_new) - g

        sy = s @ y
        Hy = H @ y
        yHy = y @ Hy

        H = H + np.outer(s, s) / sy - np.outer(Hy, Hy) / yHy
        x = x_new

    return x, hist


if __name__ == "__main__":
    x0 = np.array([2.0, 1.0])

    xs, hs = steepest_descent(x0)
    report("Steepest Descent (exact line search)", hs)

    xg, hg = gradient_descent(x0, alpha=0.4)
    report("Gradient Descent (fixed step)", hg)

    xn, hn = newton(x0)
    report("Newton", hn)

    xc, hc = conjugate_gradient(x0)
    report("Conjugate Gradient", hc)

    xd, hd = dfp(x0)
    report("DFP", hd)

    print("\n[Ground truth]")
    x_star = np.linalg.solve(Q, b)
    print("  x* =", x_star, " f(x*) =", f(x_star))
