import torch
torch.manual_seed(0)

X = torch.randn(10, 10, requires_grad=True)
Y = torch.randn(10, 10, requires_grad=True)

W = torch.randn(10, 10)
B = torch.randn(10, 10)
C = torch.randn(10, 10)

# ---- 第1问 ----
Z1 = X @ W @ X.T + B @ X + C
loss1 = Z1.sum()
loss1.backward()

gradX1 = X.grad.clone()   
print("Q1: gradX shape:", gradX1.shape)
print("Q1: gradX[0,0]:", gradX1[0,0].item())

# 清零，准备第2问
X.grad = None
Y.grad = None

# ---- 第2问 ----
Z2 = 2**X + torch.sqrt(X**2 + Y**2 + 1e-12)  # 加 eps 更稳
loss2 = Z2.sum()
loss2.backward()

gradX2 = X.grad.clone()
gradY2 = Y.grad.clone()
print("Q2: gradX shape:", gradX2.shape)
print("Q2: gradY shape:", gradY2.shape)
print("Q2: gradX[0,0]:", gradX2[0,0].item())
print("Q2: gradY[0,0]:", gradY2[0,0].item())
