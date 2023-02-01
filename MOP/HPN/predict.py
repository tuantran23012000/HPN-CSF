import torch
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import random
import torch
from matplotlib import pyplot as plt
import itertools
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse
from tools.utils import find_target, circle_points_random, get_d_paretomtl
import matplotlib as mpl
from create_pareto_front import PF
import yaml
from scalarization_function import CS_functions
from torch.autograd import Variable
from tools.utils import circle_points, sample_vec
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

def predict_2d(device,cfg,criterion,pf):
    mode = cfg['MODE']
    name = cfg['NAME']
    num_ray_init = cfg['EVAL']['Num_ray_init']
    num_ray_test = cfg['EVAL']['Num_ray_test']
    hnet1 = torch.load("./save_weights/best_weight_"+str(criterion)+"_"+str(mode)+"_"+str(name)+".pt",map_location=device)
    hnet1.eval()
    loss1 = f_1
    loss2 = f_2
    results1 = []
    targets_epo = []
    #contexts = circle_points(num_ray_test)
    contexts = np.array(sample_vec(2,num_ray_init))
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
    print(np.shape(contexts))
    contexts = rng.choice(contexts,num_ray_test)
    
    contexts = np.array([[0.2, 0.8], [0.4, 0.6],[0.3,0.7],[0.5,0.5],[0.7,0.3],[0.6,0.4],[0.9,0.1]])
    #contexts = circle_points_random([1], [25])[0]
    #print(contexts)
    for k,r in enumerate(contexts):
        r_inv = 1. / r
        ray = torch.Tensor(r.tolist()).to(device)
        ray = ray/ray.sum()
        #print(ray)
        output1 = hnet1(ray)
        l1_ = loss1(output1)
        l2_ = loss2(output1)
        losses = torch.stack([l1_, l2_])
        results1.append([l1_, l2_])
        if criterion == "Cauchy":
            target_epo = find_target(pf, criterion = criterion, context = r_inv.tolist(),cfg=cfg)
        elif criterion == "CPMTL":
            hnet1.zero_grad()
            
            l1_.backward(retain_graph = True)
            grad_ = {}
            grad_[1] = []
            
            for param in hnet1.parameters():
                if param.grad is not None:
                    grad_[1].append(Variable(param.grad.data.clone().flatten(), requires_grad = False))
            hnet1.zero_grad()
            l1_.backward(retain_graph = True)
            grad_[2] = []
            for param in hnet1.parameters():
                if param.grad is not None:
                    grad_[2].append(Variable(param.grad.data.clone().flatten(), requires_grad = False))
            grads_list = [torch.cat(grad_[1]),torch.cat(grad_[2])]
            grads = torch.stack(grads_list)
            ref_vec = torch.tensor(contexts).float()
            
            #pref_idx = np.random.randint(25)
            # loss = CS_func.get_d_paretomtl(grads,losses,ref_vec,ref_vec[pref_idx])
            target_epo = get_d_paretomtl(pf,grads,losses,ref_vec,ray)
            #print(target_epo)
        else:
            target_epo = find_target(pf, criterion = criterion, context = r.tolist(),cfg = cfg)
        targets_epo.append(target_epo)

    targets_epo = np.array(targets_epo)
    results1 = [[i[0].cpu().detach().numpy(), i[1].cpu().detach().numpy()] for i in results1]
    results1 = np.array(results1, dtype='float32')
    check = []
    MED = np.mean(np.sqrt(np.sum(np.square(targets_epo-results1),axis = 1)))
    print("MED:",MED)

    return MED, targets_epo, results1, contexts
def predict_3d(device,cfg,criterion,pf):
    mode = cfg['MODE']
    name = cfg['NAME']
    num_ray_init = cfg['TEST']['Num_ray_init']
    num_ray_test = cfg['TEST']['Num_ray_test']
    hnet1 = torch.load("./save_weights/best_weight_"+str(criterion)+"_"+str(mode)+"_"+str(name)+".pt",map_location=device)
    hnet1.eval()
    loss1 = f_1
    loss2 = f_2
    loss3 = f_3
    results1 = []
    targets_epo = []
    contexts = np.array(sample_vec(3,num_ray_init))
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
    contexts = rng.choice(contexts,num_ray_test)
    #contexts = np.array([[0.2, 0.5,0.3], [0.4, 0.25,0.35],[0.3,0.2,0.5],[0.55,0.2,0.25]])
    for r in contexts:
        r_inv = 1. / r
        ray = torch.Tensor(r.tolist()).to(device)
        output1 = hnet1(ray)
        output1 = torch.sqrt(output1)
        l1_ = loss1(output1)
        l2_ = loss2(output1)
        l3_ = loss3(output1)
        results1.append([l1_, l2_,l3_])
        if criterion == "cauchy":
            target_epo = find_target(pf, criterion = criterion, context = r_inv.tolist(),cfg=cfg)
        else:
            target_epo = find_target(pf, criterion = criterion, context = r.tolist(),cfg=cfg)
        targets_epo.append(target_epo)

    targets_epo = np.array(targets_epo)
    results1 = [[i[0].cpu().detach().numpy(), i[1].cpu().detach().numpy(),i[2].cpu().detach().numpy()] for i in results1]
    results1 = np.array(results1, dtype='float32')
    MED = np.mean(np.sqrt(np.sum(np.square(targets_epo-results1),axis = 1)))
    print("MED:",MED)

    return MED, targets_epo, results1, contexts

def draw_2d(cfg,targets_epo, results1, contexts,pf,criterion):
    mode = cfg['MODE']
    name = cfg['NAME']
    fig, ax = plt.subplots()
    for k,r in enumerate(contexts):
        r  = r/ r.sum()
        r_inv = 1. / r
        #r_inv = r
        if criterion == "KL" or criterion == "cheby" or criterion == "HVI" or criterion == "EPO":
            ep_ray = 0.9 * r_inv / np.linalg.norm(r_inv)
            ep_ray_line = np.stack([np.zeros(2), ep_ray])
            label = r'$r^{-1}$ ray' if k == 0 else ''
            ax.plot(ep_ray_line[:, 0], ep_ray_line[:, 1], color='k',
                    lw=1, ls='--', dashes=(15, 5),label=label)
            ax.arrow(.95 * ep_ray[0], .95 * ep_ray[1],
                        .05 * ep_ray[0], .05 * ep_ray[1],
                        color='k', lw=1, head_width=.02)
    ax.scatter(targets_epo[:,0], targets_epo[:,1], s=40,c='red', marker='D', alpha=1,label='Target')
    ax.scatter(results1[:, 0], results1[:, 1],s=40,c='black',marker='o',label='Predict') #HPN-PNGD
    ax.scatter(pf[:,0],pf[:,1],s=10,c='gray',label='Pareto Front',zorder=0)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.grid(color="k", linestyle="-.", alpha=0.3, zorder=0)
    ax.set_xlabel(r'$f_1$',fontsize=25)
    ax.set_ylabel(r'$f_2$',fontsize=25)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("./infer_results/"+str(name)+"_"+str(criterion)+"_"+str(mode)+".pdf")
    plt.show()

def draw_3d(cfg,targets_epo, results1, contexts,pf,criterion):
    mode = cfg['MODE']
    name = cfg['NAME']
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
    if criterion == "KL" or criterion == "cheby":
        ax.add_collection(collection)
    k = 0
    for r in contexts:
        r_inv = 1. / r
        if criterion == "KL" or criterion == "cheby":
            ep_ray = 1.0 * r_inv / np.linalg.norm(r_inv)
            ep_ray_line = np.stack([np.zeros(3), ep_ray])
            label = r'$r^{-1}$ ray' if k == 0 else ''
            k+=1
            ax.plot(ep_ray_line[:, 0], ep_ray_line[:, 1],ep_ray_line[:, 2], color='k',
                    lw=1, ls='--',label=label)
    x = results1[:,0]
    y = results1[:,1]
    z = results1[:,2]
    x_target = targets_epo[:,0]
    y_target = targets_epo[:,1]
    z_target = targets_epo[:,2]
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
    plt.tight_layout()
    plt.savefig("./infer_results/"+str(name)+"_"+str(criterion)+"_"+str(mode)+".pdf")
    plt.show()

def main(cfg,criterion,device,cpf):
    if cfg['MODE'] == '2d':
        if cfg['NAME'] == 'ex1':
            
            pf = cpf.create_pf5() 
        else:
            pf = cpf.create_pf6() 

        if cfg['EVAL']['Flag']:
            check = []
            for i in range(cfg['EVAL']['Num_eval']):
                MED, _, _, _ = predict_2d(device,cfg,criterion,pf)
                check.append(MED.tolist())
            print("Mean: ",np.array(check).mean())
            print("Std: ",np.array(check).std())
        else:
            MED, targets_epo, results1, contexts = predict_2d(device,cfg,criterion,pf)
            draw_2d(cfg,targets_epo, results1, contexts,pf,criterion)
    else:
        pf  = cpf.create_pf_3d()

        if cfg['EVAL']['Flag']:
            for i in range(cfg['EVAL']['Num_eval']):
                MED, _, _, _ = predict_3d(device,cfg,criterion,pf)
                check.append(MED.tolist())
            print("Mean: ",np.array(check).mean())
            print("Std: ",np.array(check).std())
        else:
            MED, targets_epo, results1, contexts = predict_3d(device,cfg,criterion,pf)
            draw_3d(cfg,targets_epo, results1, contexts,pf,criterion)
if __name__ == "__main__":
    device = torch.device(f"cuda:0" if torch.cuda.is_available() and not False else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/ex1.yaml', help="config file")
    parser.add_argument(
        "--solver", type=str, choices=["LS", "KL","Cheby","Utility","Cosine","Cauchy","Prod","Log","AC","MC","HV","CPMTL","EPO","HVI"], default="Utility", help="solver"
    )
    args = parser.parse_args()
    criterion = args.solver
    criterion = args.solver 
    config_file = args.config

    with open(config_file) as stream:
        cfg = yaml.safe_load(stream)
    
    if cfg['NAME'] == 'ex1':
        from problems.pb1 import f_1, f_2
        cpf = PF(f_1, f_2,None)
    elif cfg['NAME'] == 'ex2':
        from problems.pb2 import f_1, f_2
        cpf = PF(f_1, f_2,None)
    if cfg['NAME'] == 'ex3':
        from problems.pb3 import f_1, f_2, f_3
        cpf = PF(f_1, f_2, f_3)
    main(cfg,criterion,device,cpf)