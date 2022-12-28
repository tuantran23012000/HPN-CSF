import numpy as np
import torch
class PF():
    def __init__(self,f_1, f_2, f_3):
        self.f_1 = f_1
        self.f_2 = f_2
        self.f_3 = f_3
    def concave_fun_eval(self,x):
        return np.stack([self.f_1(x).item(),self.f_2(x).item()])
    def create_pf5(self):
        # example 1
        ps = np.linspace(0,1,num = 1000)
        pf = []
        for x1 in ps:
            x = torch.Tensor([[x1]])
            f= self.concave_fun_eval(x)
            pf.append(f)   
        pf = np.array(pf)
        return pf
    def create_pf6(self):
        # example 2
        ps = np.linspace(0,5,num = 1000)
        pf = []
        for x1 in ps:
            x = torch.Tensor([[x1,x1]])
            f= self.concave_fun_eval(x)
            pf.append(f)   
        pf = np.array(pf)
        return pf

    def concave_fun_eval_3d(self,x):
        """
        return the function values and gradient values
        """
        return np.stack([self.f_1(x).item(),self.f_2(x).item(),self.f_3(x).item()])

    def create_pf_3d(self):
        # example 3
        u = np.linspace(0, 1, endpoint=True, num=60)
        v = np.linspace(0, 1, endpoint=True, num=60)
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
            f= self.concave_fun_eval_3d(x)
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
