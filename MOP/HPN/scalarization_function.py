import torch

def linear_function(losses,ray):
    ls = (losses * ray).sum()
    return ls

def quadratic_function(losses,ray):
    return ray[0]*(losses[0]-2)**2 + ray[1]*(losses[1]-2)**2

def cosine_function(losses,ray):
    rl = losses * ray
    l_s = torch.sqrt((losses**2).sum())
    r_s = torch.sqrt((ray**2).sum())
    cosine = - (rl.sum()) / (l_s*r_s)
    return cosine

def utility_function(losses,ray):
    ub = 1.01
    U = 1/torch.prod((ub - losses)**ray)
    return U

def chebyshev_function(losses,ray):
    cheby = max(losses * ray)
    return cheby

def KL(losses,ray):
    m = len(losses)
    rl = torch.exp(losses * ray)
    normalized_rl = rl / (rl.sum())
    KL = (normalized_rl * torch.log(normalized_rl * m)).sum() 
    return KL

def cauchy_schwarz_function(losses,ray):
    rl = losses * ray
    l_s = (losses**2).sum()
    r_s = (ray**2).sum()
    cauchy_schwarz = 1 - ((rl.sum())**2 / (l_s*r_s))
    return cauchy_schwarz