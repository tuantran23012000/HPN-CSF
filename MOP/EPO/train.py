import sys
import os
sys.path.append(os.getcwd())
import time
import numpy as np
from matplotlib import pyplot as plt
from create_pareto_front import create_pf5,create_pf6, create_pf_3d, concave_fun_eval, concave_fun_eval_3d
import argparse
import matplotlib as mpl
from epo_search import epo_search
import torch
import itertools
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.tri import Triangulation
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
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
def find_target(pf, context=None):
    m = pf.shape[1]
    rl = context*pf
    normalized_rl = rl/np.sum(rl,axis=1).reshape(pf.shape[0],1)
    F = np.sum(normalized_rl * np.log(normalized_rl * m + 0.01),axis=1)
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

def train_2d(n=2,max_iters = 100,step_size=0.1,pf=None,num=10,name=None):

    start = time.time()
    target = []
    predict = []
    rs = np.array([[0.2, 0.8], [0.4, 0.6],[0.3,0.7],[0.5,0.5],[0.7,0.3],[0.6,0.4],[0.9,0.1]])
    contexts = np.array(sample_vec(2,num))
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
    rng = np.random.default_rng()
    #rs = rng.choice(contexts,30)
    fig, ax = plt.subplots() 
    for k, r in enumerate(rs):
        r_inv = 1. / r
        ep_ray = 0.9 * r_inv / np.linalg.norm(r_inv)
        ep_ray_line = np.stack([np.zeros(2), ep_ray])
        label = r'$r^{-1}$ ray' if k == 0 else ''
        ax.plot(ep_ray_line[:, 0], ep_ray_line[:, 1], color='k',
                lw=1, ls='--', dashes=(15, 5),label=label)
        ax.arrow(.95 * ep_ray[0], .95 * ep_ray[1],
                 .05 * ep_ray[0], .05 * ep_ray[1],
                 color='k', lw=1, head_width=.02)
        x0 = np.zeros(n)
        x0[range(0, n, 2)] = 0.1
        x0 = np.array([x0])
        sol, res = epo_search(concave_fun_eval, r=r, x=x0,step_size=step_size, max_iters=max_iters)
        target.append(find_target(pf, context=r))
        predict.append(res['ls'][-1])
    end = time.time()
    time_training = end-start
    print("Runtime training: ",time_training)
    target = np.array(target)
    predict = np.array(predict)
    MED = np.mean(np.sqrt(np.sum(np.square(target-predict),axis = 1)))
    print("MED:",MED)
    color_list = ['#28B463', '#326CAE', '#FFC300','#FF5733', 'brown']
    ax.scatter(target[:,0], target[:,1],  s=40,c='red', marker='D', alpha=1,label='Target')
    ax.scatter(predict[:, 0], predict[:, 1],s=40,c='black',marker='o',label='Predict') #HPN-PNGD
    ax.scatter(pf[:,0],pf[:,1],s=10,c='gray',label='Pareto Front',zorder=0)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.grid(color="k", linestyle="-.", alpha=0.3, zorder=0)
    ax.set_xlabel(r'$f_1$',fontsize=25)
    ax.set_ylabel(r'$f_2$',fontsize=25)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(str(name)+'_EPO_2d.pdf')
    plt.show()
    return MED

def train_3d(n=3,max_iters = 100,step_size=0.1,pf=None,num=10,name=None):
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
    midpoints = np.average(triangle_vertices, axis = 1)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    collection = Poly3DCollection(triangle_vertices,facecolors='grey', edgecolors=None)
    ax.add_collection(collection)
    start = time.time()
    target = []
    predict = []
    contexts = np.array(sample_vec(3,num))
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
    rng = np.random.default_rng()
    #rs = rng.choice(contexts,30)
    rs = np.array([[0.2, 0.5,0.3], [0.4, 0.25,0.35],[0.3,0.2,0.5],[0.55,0.2,0.25]])

    for k, r in enumerate(rs):
        r_inv = 1. / r
        ep_ray = 0.9 * r_inv / np.linalg.norm(r_inv)
        ep_ray_line = np.stack([np.zeros(3), ep_ray])
        label = r'$r^{-1}$ ray' if k == 0 else ''
        ax.plot(ep_ray_line[:, 0], ep_ray_line[:, 1],ep_ray_line[:, 2], color='k',
                lw=1, ls='--',label=label)
        x0 = np.zeros(n)
        x0[range(0, n, 2)] = 0.1
        x0[range(1, n, 2)] = 0.2
        x0[range(2, n, 2)] = 0.6
        x0 = np.array([x0])
        sol, res = epo_search(concave_fun_eval_3d, r=r, x=x0,step_size=step_size, max_iters=max_iters)
        target.append(find_target(pf, context=r))
        predict.append(res['ls'][-1])
    end = time.time()
    time_training = end-start
    print("Runtime training: ",time_training)
    target = np.array(target)
    predict = np.array(predict)
    print("MED:",np.mean(np.sqrt(np.sum(np.square(target-predict),axis = 1))))
    x = predict[:,0]
    y = predict[:,1]
    z = predict[:,2]
    x_target = target[:,0]
    y_target = target[:,1]
    z_target = target[:,2]

    ax.plot_trisurf(pf[:, 0], pf[:, 1], pf[:, 2],color='grey',alpha=0.5, shade=True,antialiased = True)
    ax.scatter(x, y, z, zdir='z',marker='o', s=10, c='black', depthshade=False,label = 'Predict')
    ax.scatter(x_target, y_target, z_target, zdir='z',marker='D', s=20, c='red', depthshade=False,label = 'Target')

    ax.set_xlabel(r'$f_1$',fontsize=18)
    ax.set_ylabel(r'$f_2$',fontsize=18)
    ax.set_zlabel(r'$f_3$',fontsize=18)
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
    ax.view_init(5, -90)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    plt.savefig(str(name)+'_EPO_3d.pdf')
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000, help="num. epochs")
    parser.add_argument("--alpha", type=float, default=0.6, help="alpha for dirichlet")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="weight decay")
    parser.add_argument("--outdim", type=int, default=3, help="output dim")
    parser.add_argument("--n_tasks", type=int, default=3, help="number of objective")
    parser.add_argument(
        "--solver", type=str, choices=["ls", "KL","cheby","utility","cosine","cauchy"], default="utility", help="solver"
    )
    parser.add_argument("--hiddendim", type=int, default=100, help="hidden_dim")
    parser.add_argument("--mode", type=str, default='2d', help="mode example")
    parser.add_argument("--name", type=str, default='ex1', help="example name")
    args = parser.parse_args()
    
    out_dim = args.outdim
    criterion = args.solver 
    hidden_dim = args.hiddendim
    lr = args.lr
    wd = args.wd
    epochs = args.epochs 
    alpha_r = args.alpha
    n_tasks = args.n_tasks
    name = args.name

    if args.mode == "2d":
        pf = create_pf5() 
        # check = []
        # for i in range(10):
        #     MED = train_2d(n=1,max_iters = 500,step_size=0.1,pf=pf,num = 500)
        #     check.append(MED)
        # print("Mean: ",np.array(check).mean())
        # print("Std: ",np.array(check).std())
        MED = train_2d(n=1,max_iters = 500,step_size=0.1,pf=pf,num = 500,name=name)
    else:
        pf  = create_pf_3d()
        train_3d(n=3,max_iters = 200,step_size=0.1,pf=pf,num = 500,name=name)
