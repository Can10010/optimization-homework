import torch

# 生成数据
torch.manual_seed(42)
num_samples = 100
X = torch.randn(num_samples, 32, 10)  # [100, 16, 20]可以随意设置
W = torch.randn(num_samples, 10, 10)  # [100, 20, 5]矩阵维度能对上就行

# 初始化参数
a = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)
optimizer = torch.optim.Adam([a, b], lr=0.01)

bits = 8
maxq = 2**bits - 1
epochs = 100
batch_size = 32

# 训练
for epoch in range(epochs):
    indices = torch.randperm(num_samples)
    total_loss = 0
    
    for i in range(0, num_samples, batch_size):
        idx = indices[i:i+batch_size]
        if len(idx) == 0: continue
        
        X_batch, W_batch = X[idx], W[idx]
        
        # 向量化计算量化参数
        W_flat = W_batch.flatten(start_dim=1)
        xmax, xmin = W_flat.max(dim=1)[0], W_flat.min(dim=1)[0]
        
        scale = (a * xmax - b * xmin) / maxq
        scale = torch.clamp(scale, min=1e-9)
        zero = -xmin / scale
        
        # 扩展维度
        scale_3d, zero_3d = scale.view(-1,1,1), zero.view(-1,1,1)
        
        # STE量化
        q_unrounded = W_batch / scale_3d + zero_3d
        q = torch.round(q_unrounded).clamp(0, maxq)
        Wq = scale_3d * (q - zero_3d) + (scale_3d * (q_unrounded - zero_3d) - scale_3d * (q - zero_3d)).detach()
        
        # 计算损失
        loss = ((X_batch @ W_batch - X_batch @ Wq) ** 2).mean()
        
        # 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            a.clamp_(0.01, 0.99)
            b.clamp_(0.01, 0.99)
        
        total_loss += loss.item() * len(idx)

print(f"a={a.item():.6f}, b={b.item():.6f}, 损失={total_loss/num_samples:.8f}")