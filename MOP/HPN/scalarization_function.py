import torch
class SC_functions():
    def __init__(self,losses,ray):
        super().__init__()
        self.losses = losses
        self.ray = ray
    def linear_function(self):
        ls = (self.losses * self.ray).sum()
        return ls

    def log_function(self):
        return (self.ray*torch.log(self.losses+1)).sum()

    def ac_function(self,rho):
        ls = (self.losses * self.ray).sum()
        cheby = max(self.losses * self.ray)
        return cheby + rho*ls
    
    def mc_function(self,rho):
        ls = (self.losses * self.ray).sum()
        cheby = max(self.losses * self.ray + rho*ls)
        return cheby
    
    def hv_function(self,dynamic_weight,rho):
        rl = self.losses * self.ray
        l_s = torch.norm(self.losses)
        r_s = torch.norm(self.ray)
        cosine = - (rl.sum()) / (l_s*r_s) 
        hv = -(dynamic_weight*self.losses).sum()  + rho * cosine
        return hv

    def product_function(self):
        return torch.prod((self.losses+1)**self.ray)

    def cosine_function(self):
        rl =self.losses * self.ray
        l_s = torch.sqrt((self.losses**2).sum())
        r_s = torch.sqrt((self.ray**2).sum())
        cosine = - (rl.sum()) / (l_s*r_s)
        return cosine

    def utility_function(self,ub):
        
        U = 1/torch.prod((ub - self.losses)**self.ray)
        return U

    def chebyshev_function(self):
        cheby = max(self.losses * self.ray)
        return cheby

    def KL_function(self):
        m = len(self.losses)
        rl = torch.exp(self.losses * self.ray)
        normalized_rl = rl / (rl.sum())
        KL = (normalized_rl * torch.log(normalized_rl * m)).sum() 
        return KL

    def cauchy_schwarz_function(self):
        rl = self.losses * self.ray
        l_s = (self.losses**2).sum()
        r_s = (self.ray**2).sum()
        cauchy_schwarz = 1 - ((rl.sum())**2 / (l_s*r_s))
        return cauchy_schwarz