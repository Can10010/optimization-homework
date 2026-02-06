import torch

M = torch.randn(10, 20)      
print(M.shape)

m_max = M.max()
m_min = M.min()
m_mean = M.mean()
m_var = M.var(unbiased=False)    

m_norm = torch.linalg.norm(M)    

print("max:", m_max.item())
print("min:", m_min.item())
print("mean:", m_mean.item())
print("var:", m_var.item())
print("norm:", m_norm.item())

row_max = M.max(dim=1).values
row_min = M.min(dim=1).values
row_mean = M.mean(dim=1)
row_var = M.var(dim=1, unbiased=False)
row_norm = torch.linalg.norm(M, dim=1)   

print("row_max shape:", row_max.shape)   
print("row_norm shape:", row_norm.shape) 

col_max = M.max(dim=0).values
col_min = M.min(dim=0).values
col_mean = M.mean(dim=0)
col_var = M.var(dim=0, unbiased=False)
col_norm = torch.linalg.norm(M, dim=0)  

print("col_max shape:", col_max.shape)  
print("col_norm shape:", col_norm.shape) 

U, S, Vh = torch.linalg.svd(M, full_matrices=False)
print("U:", U.shape)    
print("S:", S.shape)     
print("Vh:", Vh.shape)  

A = float(input("请输入阈值 A: "))

mask = M > A
indices = torch.nonzero(mask, as_tuple=False)  
values = M[mask]                              

print("count:", values.numel())
print("indices shape:", indices.shape)

k = min(10, values.numel())
for i in range(k):
    r, c = indices[i].tolist()
    v = values[i].item()
    print(f"M[{r},{c}] = {v:.6f}")
