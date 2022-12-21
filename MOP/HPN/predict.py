import torch
import os
import sys
sys.path.append(os.getcwd())
import logging
import numpy as np
import random
import torch
from matplotlib import pyplot as plt
from problems import f_1, f_2,f_3
from create_pareto_front import create_pf5,create_pf6, create_pf_3d
import itertools
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse
def simplex(n_vals):
    base = np.linspace(0, 0.25, n_vals, endpoint=False)
    coords = np.asarray(list(itertools.product(base, repeat=3)))
    return coords[np.isclose(coords.sum(axis=-1), 0.25)]

color_list = ['#28B463', '#326CAE', '#FFC300','#FF5733', 'brown']
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 18,
         }
font_legend = {'family': 'Times New Roman',
               'weight': 'normal',
                'size': 18,
               }
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

set_logger()

#device = torch.device(f"cuda:0" if torch.cuda.is_available() and not False else "cpu")
device = torch.device("cpu")
def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = np.pi / 20. if min_angle is None else min_angle
    ang1 = np.pi * 9 / 20. if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K, endpoint=True)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y]


def find_target(pf, criterion, context=None):
    if criterion == 'cheby':
        F = np.max(context*pf,axis = 1)
        return pf[F.argmin(), :]
    elif criterion == 'ls':
        F = context[0]*pf[:, 0] + context[1]*pf[:, 1]
        return pf[F.argmin(), :]
    elif criterion == 'utility':
        ub = 1.01
        F = 1/np.prod(((ub-pf)**context),axis=1)
        return pf[F.argmin(), :]
    elif criterion == 'cosine':
        rl = np.sum(context*pf,axis = 1)
        l_s = np.sqrt(np.sum(pf**2,axis = 1))
        r_s = np.sqrt(np.sum(np.array(context)**2))
        F = - (rl) / (l_s*r_s)
        return pf[F.argmin(), :]
    elif criterion == 'KL':
        m = pf.shape[1]
        rl = np.exp(context*pf)
        normalized_rl = rl/np.sum(rl,axis=1).reshape(pf.shape[0],1)
        F = np.sum(normalized_rl * np.log(normalized_rl * m),axis=1) 
        return pf[F.argmin(),:]
    elif criterion == 'cauchy':
        rl = np.sum(context*pf,axis = 1)
        l_s = np.sum(pf**2,axis = 1)
        r_s = np.sum(np.array(context)**2)
        F = 1 - (rl)**2 / (l_s*r_s)
        return pf[F.argmin(),:]

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

def predict_2d(criterion,num_ray,mode,name):
    pf = create_pf5() # pareto for OF01
    hnet1 = torch.load("./save_weights/best_weight_"+str(criterion)+"_"+str(mode)+"_"+str(name)+".pt")
    hnet1.eval()
    loss1 = f_1
    loss2 = f_2
    results1 = []
    fig, ax = plt.subplots() 
    targets_epo = []
    #contexts = circle_points(num_ray)
    #contexts = np.array(sample_vec(2,num_ray))
    contexts = np.array([[0.2, 0.8], [0.4, 0.6],[0.3,0.7],[0.5,0.5],[0.7,0.3],[0.6,0.4],[0.9,0.1]])
    for k,r in enumerate(contexts):
        r_inv = 1. / r
        if criterion == "KL" or criterion == "cheby":
            ep_ray = 0.9 * r_inv / np.linalg.norm(r_inv)
            ep_ray_line = np.stack([np.zeros(2), ep_ray])
            label = r'$r^{-1}$ ray' if k == 0 else ''
            ax.plot(ep_ray_line[:, 0], ep_ray_line[:, 1], color='k',
                    lw=1, ls='--', dashes=(15, 5),label=label)
            ax.arrow(.95 * ep_ray[0], .95 * ep_ray[1],
                        .05 * ep_ray[0], .05 * ep_ray[1],
                        color='k', lw=1, head_width=.02)
        ray = torch.Tensor(r.tolist()).to(device)
        output1 = hnet1(ray)
        l1_ = loss1(output1)
        l2_ = loss2(output1)
        results1.append([l1_, l2_])
        if criterion == "cauchy":
            target_epo = find_target(pf, criterion = criterion, context = r_inv.tolist())
        else:
            target_epo = find_target(pf, criterion = criterion, context = r.tolist())
        targets_epo.append(target_epo)

    targets_epo = np.array(targets_epo)
    results1 = [[i[0].cpu().detach().numpy(), i[1].cpu().detach().numpy()] for i in results1]
    results1 = np.array(results1, dtype='float32')
    print("MED:",np.mean(np.sqrt(np.sum(np.square(targets_epo-results1),axis = 1))))
    ax.xaxis.set_label_coords(1.015, -0.03)
    ax.yaxis.set_label_coords(-0.01, 1.01)
    ax.scatter(targets_epo[:,0], targets_epo[:,1], s=60,c=color_list[0], marker='o', alpha=1,label='Target')
    ax.scatter(results1[:, 0], results1[:, 1],s=40,c=color_list[2],marker='D',label='Predict') #HPN-PNGD
    ax.scatter(pf[:,0],pf[:,1],s=10,c='gray',label='Pareto Front',zorder=0)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.grid(color="k", linestyle="-.", alpha=0.3, zorder=0)
    ax.set_xlabel(r'$f_1$')
    ax.set_ylabel(r'$f_2$')
    ax.legend()
    plt.savefig("./infer_results/"+str(name)+"_"+str(criterion)+"_"+str(mode)+".png")
    plt.show()

def predict_3d(criterion,num_ray,mode,name):
    pf = create_pf_3d()
    hnet1 = torch.load("./save_weights/best_weight_"+str(criterion)+"_"+str(mode)+"_"+str(name)+".pt",map_location=torch.device('cpu'))
    hnet1.eval()
    loss1 = f_1
    loss2 = f_2
    loss3 = f_3
    results1 = []
    sim = simplex(5)
    x = sim[:, 0]
    y = sim[:, 1]
    z = sim[:, 2]
    tri = Triangulation(x, y)
    triangle_vertices = np.array([np.array([[x[T[0]], y[T[0]], z[T[0]]],
                                            [x[T[1]], y[T[1]], z[T[1]]], 
                                            [x[T[2]], y[T[2]], z[T[2]]]]) for T in tri.triangles])
    triangle_vertices = np.append(triangle_vertices,np.array([np.array([[1,0,0],[0.8,0,0.2],[0.8,0.2,0]])/4]),axis=0)
    triangle_vertices = np.append(triangle_vertices,np.array([np.array([[0,1,0],[0,0.8,0.2],[0.2,0.8,0]])/4]),axis=0)
    triangle_vertices = np.append(triangle_vertices,np.array([np.array([[0,0,1],[0.2,0,0.8],[0,0.2,0.8]])/4]),axis=0)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    collection = Poly3DCollection(triangle_vertices,facecolors='grey', edgecolors=None)
    ax.add_collection(collection)
    targets_epo = []
    contexts = np.array(sample_vec(3,num_ray))
    tmp = []
    for r in contexts:
        flag = True
        for i in r:
            if i <=0.16:
                flag = False
                break
        if flag:

            tmp.append(r)
    contexts = np.array(tmp)
    contexts = np.array([[0.2, 0.5,0.3], [0.4, 0.25,0.35],[0.3,0.2,0.5],[0.55,0.2,0.25]])
    for r in contexts:
        r_inv = 1. / r
        if criterion == "KL" or criterion == "cheby":
            ep_ray = 1.1 * r_inv / np.linalg.norm(r_inv)
            ep_ray_line = np.stack([np.zeros(3), ep_ray])
            ax.plot(ep_ray_line[:, 0], ep_ray_line[:, 1],ep_ray_line[:, 2], color='k',
                    lw=1, ls='--')
        ray = torch.Tensor(r.tolist()).to(device)
        output1 = hnet1(ray)
        output1 = torch.sqrt(output1)
        l1_ = loss1(output1)
        l2_ = loss2(output1)
        l3_ = loss3(output1)
        results1.append([l1_, l2_,l3_])
        if criterion == "cauchy":
            target_epo = find_target(pf, criterion = criterion, context = r_inv.tolist())
        else:
            target_epo = find_target(pf, criterion = criterion, context = r.tolist())
        targets_epo.append(target_epo)

    targets_epo = np.array(targets_epo)
    results1 = [[i[0].cpu().detach().numpy(), i[1].cpu().detach().numpy(),i[2].cpu().detach().numpy()] for i in results1]
    results1 = np.array(results1, dtype='float32')
    print("MED:",np.mean(np.sqrt(np.sum(np.square(targets_epo-results1),axis = 1))))
    x = results1[:,0]
    y = results1[:,1]
    z = results1[:,2]
    x_target = targets_epo[:,0]
    y_target = targets_epo[:,1]
    z_target = targets_epo[:,2]
    ax.plot_trisurf(pf[:, 0], pf[:, 1], pf[:, 2],color='grey',alpha=0.5, shade=True,antialiased = True)
    ax.scatter(x, y, z, zdir='z',marker='o', s=10, c='black', depthshade=False,label = 'Predict')
    ax.scatter(x_target, y_target, z_target, zdir='z',marker='D', s=20, c='red', depthshade=False,label = 'Target')
    ax.legend()
    ax.set_xlabel(r'$f_1$')
    ax.set_ylabel(r'$f_2$')
    ax.set_zlabel(r'$f_3$')
    ax.grid(True)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_zticks([0.2, 0.4, 0.6, 0.8])
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.zaxis.set_rotate_label(False)
    ax.xaxis.set_rotate_label(False)
    [t.set_va('bottom') for t in ax.get_yticklabels()]
    [t.set_ha('center') for t in ax.get_yticklabels()]
    [t.set_va('bottom') for t in ax.get_xticklabels()]
    [t.set_ha('center') for t in ax.get_xticklabels()]
    [t.set_va('center') for t in ax.get_zticklabels()]
    [t.set_ha('center') for t in ax.get_zticklabels()]
    ax.xaxis._axinfo['tick']['inward_factor'] = 0
    ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.yaxis._axinfo['tick']['inward_factor'] = 0
    ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.zaxis._axinfo['tick']['inward_factor'] = 0
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.view_init(elev=15., azim=100.)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    plt.savefig("./infer_results/"+str(name)+"_"+str(criterion)+"_"+str(mode)+".png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--criterion", type=str, choices=["ls", "KL","cheby","utility","cosine","cauchy"], default="ls", help="solver"
    )
    parser.add_argument("--num_ray", type=int, default=10, help="hidden_dim")
    parser.add_argument("--mode", type=str, default='2d', help="mode example")
    parser.add_argument("--name", type=str, default='ex1', help="example name")
    args = parser.parse_args()
    criterion = args.criterion
    num_ray = args.num_ray
    mode = args.mode
    name = args.name
    if mode == "2d":
        predict_2d(criterion,num_ray,mode,name)
    else:
        predict_3d(criterion,num_ray,mode,name)
