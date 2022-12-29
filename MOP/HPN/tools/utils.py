import logging
import random
import torch
import numpy as np
from tools.hv import HvMaximization
def set_logger():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def sample_vec(n,m):
    vector = [0]*n
    unit = np.linspace(0, 1, m)
    rays = []
    def sample(i, sum):
        if i == n-1:
            vector[i] = 1-sum
            rays.append(vector.copy())
            return vector
        for value in unit:
            if value > 1-sum:
                break
            else:
                vector[i] = value
                sample(i+1, sum+value)
    sample(0,0)
    return rays

def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = np.pi / 20. if min_angle is None else min_angle
    ang1 = np.pi * 9 / 20. if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K, endpoint=True)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y]

def find_target(pf, criterion, context,cfg):

    if criterion == 'Log':
        F = np.sum(context*np.log(pf+1),axis = 1)

    elif criterion == 'Prod':
        F = np.prod((pf+1)**context,axis = 1)

    elif criterion == 'AC':
        F1 = np.max(context*pf,axis = 1)
        F2 = np.sum(context*pf,axis = 1)
        rho = cfg['TRAIN']['Solver'][criterion]['Rho']
        F = F1 + rho*F2

    elif criterion == 'MC':
        rho = cfg['TRAIN']['Solver'][criterion]['Rho']
        F1 = np.sum(context*pf,axis = 1).reshape(pf.shape[0],1)
        F = np.max(context*pf + rho*F1,axis = 1)

    elif criterion == 'HV':
        n_mo_obj = cfg['TRAIN']['N_task']
        ref_point = tuple(map(int, cfg['TRAIN']['Ref_point'].split(',')))
        rho = cfg['TRAIN']['Solver'][criterion]['Rho'] 
        mo_opt = HvMaximization(1, n_mo_obj, ref_point)
        loss_numpy = pf[:, :,np.newaxis]
        n_samples = loss_numpy.shape[0]
        dynamic_weight = []
        for i_sample in range(0, n_samples):
            dynamic_weight.append((mo_opt.compute_weights(loss_numpy[i_sample,:,:])).reshape(1,n_mo_obj).tolist()[0])
        dynamic_weight = np.array(dynamic_weight)
        rl = np.sum(context*pf,axis = 1)
        l_s = np.sqrt(np.sum(pf**2,axis = 1))
        r_s = np.sqrt(np.sum(np.array(context)**2))
        cosine = - (rl) / (l_s*r_s)
        F = -np.sum((dynamic_weight*pf),axis =1) + rho*cosine

    elif criterion == 'Cheby':
        F = np.max(context*pf,axis = 1)

    elif criterion == 'LS':
        F = np.sum(context*pf,axis = 1)

    elif criterion == 'Utility':
        ub = cfg['TRAIN']['Solver'][criterion]['Ub']
        F = 1/np.prod(((ub-pf)**context),axis=1)

    elif criterion == 'Cosine':
        rl = np.sum(context*pf,axis = 1)
        l_s = np.sqrt(np.sum(pf**2,axis = 1))
        r_s = np.sqrt(np.sum(np.array(context)**2))
        F = - (rl) / (l_s*r_s)

    elif criterion == 'KL':
        m = pf.shape[1]
        rl = np.exp(context*pf)
        normalized_rl = rl/np.sum(rl,axis=1).reshape(pf.shape[0],1)
        F = np.sum(normalized_rl * np.log(normalized_rl * m),axis=1) 

    elif criterion == 'Cauchy':
        rl = np.sum(context*pf,axis = 1)
        l_s = np.sum(pf**2,axis = 1)
        r_s = np.sum(np.array(context)**2)
        F = 1 - (rl)**2 / (l_s*r_s)

    return pf[F.argmin(), :]