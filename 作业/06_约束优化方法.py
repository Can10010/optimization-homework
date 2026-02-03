import numpy as np

# 初始值
x1, x2 = 0, 0  

lambda_ = 50      # 惩罚系数
learning_rate = 0.01  # 学习率

for i in range(50):
    grad_x1 = 2*x1 + 2*lambda_*(x1-1)
    grad_x2 = 2*x2
    
    x1 = x1 - learning_rate * grad_x1
    x2 = x2 - learning_rate * grad_x2

print(f"x1 = {x1}")
print(f"x2 = {x2}")
print(f"目标函数值 = {x1**2 + x2**2}")
