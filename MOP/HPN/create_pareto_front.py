import numpy as np
import torch
from problems import f_1, f_2, f_3
from autograd import grad
def concave_fun_eval(x):
    f1_dx = grad(f_1)
    f2_dx = grad(f_2)
    return np.stack([f_1(x).item(),f_2(x).item()]), np.stack([f1_dx(x), f2_dx(x)])
def create_pf5():
    # example 1
    ps = np.linspace(0,1,num = 1000)
    pf = []
    for x1 in ps:
        x = torch.Tensor([[x1]])
        f,_ = concave_fun_eval(x)
        pf.append(f)   
    pf = np.array(pf)
    return pf
def create_pf6():
    # example 2
    ps = np.linspace(0,5,num = 1000)
    pf = []
    for x1 in ps:
        x = torch.Tensor([[x1,x1]])
        f,_ = concave_fun_eval(x)
        pf.append(f)   
    pf = np.array(pf)
    return pf

def concave_fun_eval_3d(x):
    """
    return the function values and gradient values
    """
    f1_dx = grad(f_1)
    f2_dx = grad(f_2)
    f3_dx = grad(f_3)
    return np.stack([f_1(x).item(), f_2(x).item(), f_3(x).item()]), np.stack([f1_dx(x), f2_dx(x), f3_dx(x)])

def create_pf_3d():
    # example 3
    u = np.linspace(0, 1, endpoint=True, num=10)
    v = np.linspace(0, 1, endpoint=True, num=10)
    tmp = []
    for i in u:
        for j in v:
            if 1-i**2-j**2 >=0:
                tmp.append([np.sqrt(1-i**2-j**2),i,j])
                tmp.append([i,np.sqrt(1-i**2-j**2),j])
                tmp.append([i,j,np.sqrt(1-i**2-j**2)])
    uv = np.array(tmp)
    print(f"uv.shape={uv.shape}")
    ls = []
    for x in uv:
        x = torch.Tensor([x])
        f,_ = concave_fun_eval_3d(x)
        ls.append(f)
    ls = np.stack(ls)
    po, pf = [], []
    for i, x in enumerate(uv):
        l_i = ls[i]
        po.append(x)
        pf.append(l_i)
    po = np.stack(po)
    pf = np.stack(pf)
    return pf
