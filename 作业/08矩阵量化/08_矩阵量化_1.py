import torch
import numpy as np

# 生成100个矩阵
matrices = [torch.randn(10, 10) for _ in range(100)]
bits = 8
maxq = 2**bits - 1

# 黄金分割法函数
def golden_search(func, L, U, tol=1e-6):
    r = (np.sqrt(5) - 1) / 2  # 0.618
    
    x1 = U - r * (U - L)
    x2 = L + r * (U - L)
    f1 = func(x1)
    f2 = func(x2)
    
    for _ in range(100):
        if f1 < f2:
            U = x2
            x2 = x1
            f2 = f1
            x1 = U - r * (U - L)
            f1 = func(x1)
        else:
            L = x1
            x1 = x2
            f1 = f2
            x2 = L + r * (U - L)
            f2 = func(x2)
        
        if (U - L) < tol:
            break
    
    return (L + U) / 2

# 对每个矩阵优化
results = []
for i, w in enumerate(matrices):
    # 计算损失的函数
    def compute_loss_wrapper(a, b):
        xmax = torch.max(w)
        xmin = torch.min(w)
        scale = (a * xmax - b * xmin) / maxq
        scale = torch.clamp(scale, min=1e-9)
        zero = -xmin / scale
        q = torch.clamp(torch.round(w / scale + zero), 0, maxq)
        w_recon = scale * (q - zero)
        loss = torch.mean((w_recon - w)**2)
        return loss.item()
    
    # 交替优化
    a_opt = golden_search(lambda a_val: compute_loss_wrapper(a_val, 0.5), 0.01, 0.99)
    b_opt = golden_search(lambda b_val: compute_loss_wrapper(a_opt, b_val), 0.01, 0.99)
    
    # 计算最终损失
    final_loss = compute_loss_wrapper(a_opt, b_opt)
    
    results.append([a_opt, b_opt, final_loss])

# 计算平均结果
avg_a = np.mean([r[0] for r in results])
avg_b = np.mean([r[1] for r in results])
avg_loss = np.mean([r[2] for r in results])

print(f"平均 a: {avg_a:.4f}")
print(f"平均 b: {avg_b:.4f}")
print(f"平均损失: {avg_loss:.8f}")