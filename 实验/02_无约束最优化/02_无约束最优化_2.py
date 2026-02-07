import numpy as np

cnt_f = 0
cnt_g = 0


def f(x):
    global cnt_f
    cnt_f += 1
    x1, x2, x3 = x
    return x1**2 + 3*np.sin(x2) - x1*(x2**2)*(x3**2)


def grad_f(x):
    global cnt_g
    cnt_g += 1
    x1, x2, x3 = x
    g1 = 2*x1 - (x2**2)*(x3**2)
    g2 = 3*np.cos(x2) - 2*x1*x2*(x3**2)
    g3 = -2*x1*(x2**2)*x3
    return np.array([g1, g2, g3])


def line_search(x, p, g, alpha0=1.0, rho=0.5, c1=1e-4):
    alpha = alpha0
    fx = f(x)
    while f(x + alpha*p) > fx + c1*alpha*np.dot(g, p):
        alpha *= rho
    return alpha


def dfp_quasi_newton(x0, tol=1e-6, max_iter=100):
    x = x0.copy()
    n = len(x)
    H = np.eye(n)          # 初始逆 Hessian
    g = grad_f(x)

    k = 0
    while np.linalg.norm(g) > tol and k < max_iter:
        p = -H @ g                     # 搜索方向
        alpha = line_search(x, p, g)   # 线搜索
        x_new = x + alpha*p
        g_new = grad_f(x_new)

        s = x_new - x
        y = g_new - g

        # DFP 更新
        sy = np.dot(s, y)
        if sy > 1e-12:  # 防止数值问题
            Hy = H @ y
            H = H + np.outer(s, s)/sy - np.outer(Hy, Hy)/(y @ Hy)

        x, g = x_new, g_new
        k += 1

    return x, f(x), k


if __name__ == "__main__":
    x0 = np.array([1.0, -1.0, 0.0])   # 题目给定初值

    xmin, fmin, iters = dfp_quasi_newton(x0)

    print("收敛点 x* =", xmin)
    print("最小值 f(x*) =", fmin)
    print("迭代次数 =", iters)
    print("函数计算次数 =", cnt_f)
    print("梯度计算次数 =", cnt_g)
