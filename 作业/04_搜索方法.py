import numpy as np

rng = np.random.default_rng(0)
a = rng.random() + 0.2    
b = rng.normal()

def f(x):
    return (x - 2)/x - a*(x - 1) - b

def fp(x):
    return -a + 2/(x**2)

# ========== 1) 找单谷区间 [L, U]（在负半轴） ==========
U = -1e-3
L = -1.0

while fp(L) > 0:
    L *= 2.0  
print("a =", a, "b =", b)
print("unimodal interval [L, U] =", (L, U), "fp(L)=", fp(L), "fp(U)=", fp(U))

# ========== 2) 方法1：导数二分法（在 fp 上找根） ==========
def bisection_on_derivative(L, U, tol=1e-8, max_iter=200):
    l, r = L, U
    for _ in range(max_iter):
        m = (l + r)/2
        if abs(r - l) < tol:
            break
        if fp(m) > 0:
            r = m
        else:
            l = m
    x_star = (l + r)/2
    return x_star

# ========== 2) 方法2：黄金分割法（只用 f 值） ==========
def golden_section(L, U, tol=1e-8, max_iter=300):
    phi = (np.sqrt(5) - 1) / 2 
    l, r = L, U
    x1 = r - phi*(r - l)
    x2 = l + phi*(r - l)
    f1 = f(x1)
    f2 = f(x2)

    for _ in range(max_iter):
        if abs(r - l) < tol:
            break
        if f1 > f2:
            l = x1
            x1 = x2
            f1 = f2
            x2 = l + phi*(r - l)
            f2 = f(x2)
        else:
            r = x2
            x2 = x1
            f2 = f1
            x1 = r - phi*(r - l)
            f1 = f(x1)

    return (l + r)/2

# ========== 2) 方法3：抛物线法（三点二次插值） ==========
def parabolic_interpolation(L, U, tol=1e-8, max_iter=200):
    x0 = L
    x2 = U
    x1 = (L + U)/2

    f0, f1, f2 = f(x0), f(x1), f(x2)

    for _ in range(max_iter):
        denom = (x1 - x0)*(f1 - f2) - (x1 - x2)*(f1 - f0)
        if abs(denom) < 1e-18:
            break

        numer = (x1 - x0)**2*(f1 - f2) - (x1 - x2)**2*(f1 - f0)
        x3 = x1 - 0.5 * numer / denom

        x3 = min(max(x3, L), U)
        if x3 >= -1e-12:
            x3 = -1e-6

        f3 = f(x3)

        if x3 < x1:
            if f3 < f1:
                x2, f2 = x1, f1
                x1, f1 = x3, f3
            else:
                x0, f0 = x3, f3
        else:
            if f3 < f1:
                x0, f0 = x1, f1
                x1, f1 = x3, f3
            else:
                x2, f2 = x3, f3

        if abs(x2 - x0) < tol:
            break

    return x1

x_bis = bisection_on_derivative(L, U)
x_gld = golden_section(L, U)
x_par = parabolic_interpolation(L, U)

print("\n[bisection] x* =", x_bis, "f(x*) =", f(x_bis))
print("[golden]    x* =", x_gld, "f(x*) =", f(x_gld))
print("[parabola]  x* =", x_par, "f(x*) =", f(x_par))
