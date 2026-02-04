import torch
import numpy as np

# 生成数据
torch.manual_seed(42)
num_samples = 100
X = torch.randn(num_samples, 32, 10)
W = torch.randn(num_samples, 10, 10)
bits = 8
maxq = 2**bits - 1

def compute_loss(a, b):
    """计算给定a,b参数时的损失"""
    total_loss = 0
    
    with torch.no_grad():
        for i in range(num_samples):
            X_i, W_i = X[i], W[i]
            
            # 计算量化参数
            xmax, xmin = torch.max(W_i), torch.min(W_i)
            scale = (a * xmax.item() - b * xmin.item()) / maxq
            scale = max(scale, 1e-9)
            zero = -xmin.item() / scale
            
            # 量化-反量化
            q = torch.clamp(torch.round(W_i / scale + zero), 0, maxq)
            Wq = scale * (q - zero)
            
            # 计算损失
            loss = torch.mean((X_i @ W_i - X_i @ Wq) ** 2)
            total_loss += loss.item()
    
    return total_loss / num_samples

# PSO参数
num_particles = 20
dimensions = 2  # a和b
max_iter = 50
bounds = (0.01, 0.99)

# 初始化粒子
positions = np.random.uniform(bounds[0], bounds[1], (num_particles, dimensions))
velocities = np.random.uniform(-0.1, 0.1, (num_particles, dimensions))
pbest_positions = positions.copy()
pbest_fitness = np.full(num_particles, np.inf)
gbest_position = None
gbest_fitness = np.inf

# PSO参数
w = 0.8    # 惯性权重
c1 = 1.5   # 个体学习因子
c2 = 1.5   # 社会学习因子

# PSO优化
for iter_num in range(max_iter):
    # 评估所有粒子
    for i in range(num_particles):
        fitness = compute_loss(positions[i, 0], positions[i, 1])
        
        # 更新个体最佳
        if fitness < pbest_fitness[i]:
            pbest_fitness[i] = fitness
            pbest_positions[i] = positions[i].copy()
        
        # 更新全局最佳
        if fitness < gbest_fitness:
            gbest_fitness = fitness
            gbest_position = positions[i].copy()
    
    # 更新粒子的速度和位置
    for i in range(num_particles):
        r1, r2 = np.random.random(dimensions), np.random.random(dimensions)
        
        # 速度更新
        velocities[i] = (
            w * velocities[i] + 
            c1 * r1 * (pbest_positions[i] - positions[i]) + 
            c2 * r2 * (gbest_position - positions[i])
        )
        
        # 限制速度
        velocities[i] = np.clip(velocities[i], -0.2, 0.2)
        
        # 位置更新
        positions[i] += velocities[i]
        positions[i] = np.clip(positions[i], bounds[0], bounds[1])

# 最终结果
print(f"最佳 a: {gbest_position[0]:.6f}")
print(f"最佳 b: {gbest_position[1]:.6f}")
print(f"最佳损失: {gbest_fitness:.8f}")