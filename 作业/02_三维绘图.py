# 02_三维绘图.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  

def surface_plot(ax, X, Y, Z, title=""):
    ax.plot_surface(X, Y, Z, rstride=2, cstride=2, linewidth=0, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

x = np.linspace(-5, 5, 300)
y = np.linspace(-5, 5, 300)
X, Y = np.meshgrid(x, y)

# =========================
# (1) 画题目给的 z 的三维曲面
# =========================
Z1 = (1 - (1/5)*X + X**5 + (1/2)*Y**3) / (2 ** (3*X**2 + 2*Y**3)) 
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection="3d")
surface_plot(ax1, X, Y, Z1, title="(1) 曲面：题目1")

# =========================
# (2) 画 z = sin(sqrt(x^2 + y^2)) 的三维曲面
# =========================
R = np.sqrt(X**2 + Y**2)
Z2 = np.sin(R)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection="3d")
surface_plot(ax2, X, Y, Z2, title="(2) 曲面：z = sin(sqrt(x^2 + y^2))")

# =========================
# (3) 同一三维坐标下画多个平面
# =========================
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection="3d")
x3 = np.linspace(-10, 10, 80)
y3 = np.linspace(-10, 10, 80)
X3, Y3 = np.meshgrid(x3, y3)
Zp1 = -(5*X3 + 8*Y3) / 3          
Zp2 = (13*X3 - 5*Y3) / 4         
Zp3 = (-X3 + 10*Y3) / 5         
Zp4 = np.full_like(X3, 20.0)    

ax3.plot_surface(X3, Y3, Zp1, alpha=0.45)
ax3.plot_surface(X3, Y3, Zp2, alpha=0.45)
ax3.plot_surface(X3, Y3, Zp3, alpha=0.45)
ax3.plot_surface(X3, Y3, Zp4, alpha=0.25)

ax3.set_title("(3) 多个平面（同一 3D 坐标）")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_zlabel("z")

ax3.view_init(elev=20, azim=35)

plt.show()