import torch

# 生成训练数据
X_list = [torch.randn(32, 10) for _ in range(100)]  # 100个X矩阵
W_list = [torch.randn(10, 10) for _ in range(100)]  # 100个W矩阵

bits = 8
maxq = 2**bits - 1

# 初始化可优化参数
a = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)

# 优化器
optimizer = torch.optim.Adam([a, b], lr=0.01)

# 训练循环
for epoch in range(500):
    total_loss = 0
    
    for X, W in zip(X_list, W_list):
        # 计算量化参数
        xmax = torch.max(W)
        xmin = torch.min(W)
        
        # 使用a,b调节的Scale公式
        scale = (a * xmax - b * xmin) / maxq
        scale = torch.clamp(scale, min=1e-9)
        
        zero = -xmin / scale
        
        # 使用Straight-Through Estimator (STE)处理round操作
        # 前向传播使用round，反向传播绕过round
        q_unrounded = W / scale + zero
        q = torch.round(q_unrounded)
        q = torch.clamp(q, 0, maxq)
        
        # STE技巧：前向使用q，反向使用q_unrounded
        Wq = scale * (q - zero)
        Wq = Wq + (scale * (q_unrounded - zero) - Wq).detach()
        
        # 计算损失
        loss = torch.mean((X @ W - X @ Wq) ** 2)
        total_loss += loss
    
    avg_loss = total_loss / len(X_list)
    
    optimizer.zero_grad()
    avg_loss.backward()
    optimizer.step()
    
    # 限制参数范围
    with torch.no_grad():
        a.clamp_(0.01, 0.99)
        b.clamp_(0.01, 0.99)

print(f"a: {a.item():.6f}")
print(f"b: {b.item():.6f}")
print(f"损失: {avg_loss.item():.8f}")

