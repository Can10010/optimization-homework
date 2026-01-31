import torch

dtype = torch.double

A  = torch.randn(10, dtype=dtype)
B  = torch.randn(10, dtype=dtype)
C  = torch.randn(10, dtype=dtype)
X0 = torch.randn(10, dtype=dtype)

def f(x: torch.Tensor) -> torch.Tensor:
    quad  = (A * x * x).sum()    
    lin   = (B * x).sum()        
    const = C.sum()              
    norm2 = (x * x).sum()         

    s = quad + lin + const + norm2
    return torch.log(s)           

H = torch.autograd.functional.hessian(f, X0)
print(H)
