import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 500)
y = np.linspace(-5, 5, 500)
X, Y = np.meshgrid(x, y)

Z = (1 - (1/5)*X + X**5 + (1/2)*Y**3) * (1 / (2 ** (3*X**2 + 2*Y**3)))

levels = np.linspace(-1, 1, 20)
cs = plt.contour(X, Y, Z, levels=levels)
plt.clabel(cs, inline=True, fontsize=8)

plt.xlim(-5, 5)
plt.ylim(-5, 5)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Contour demo")

plt.show()