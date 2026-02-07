import numpy as np
import matplotlib.pyplot as plt

x = np.array([120, 92, 90, 80, 72, 60], dtype=float)
y = np.array([ 90, 84, 83, 80, 75, 68], dtype=float)

coef2 = np.polyfit(x, y, 2)
coef3 = np.polyfit(x, y, 3)

p2 = np.poly1d(coef2)
p3 = np.poly1d(coef3)

print("二次拟合：y =", p2)
print("三次拟合：y =", p3)

x_fit = np.linspace(x.min(), x.max(), 400)
y2_fit = p2(x_fit)
y3_fit = p3(x_fit)

plt.scatter(x, y, label="origin data")         
plt.plot(x_fit, y2_fit, label="polyfit deg=2") 
plt.plot(x_fit, y3_fit, label="polyfit deg=3") 

plt.xlabel("x-price")
plt.ylabel("y-amount")
plt.title("polyfit")
plt.grid(True)
plt.legend()
plt.show()
