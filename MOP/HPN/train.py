import sys
import os
sys.path.append(os.getcwd())
from torch import nn
import logging
import time
import random
from tqdm import tqdm
from models import Toy_Hypernetwork_2d, Toy_Hypernetwork_3d
import numpy as np
import torch
from matplotlib import pyplot as plt
from problems import f_1, f_2, f_3
from create_pareto_front import create_pf5,create_pf6, create_pf_3d
from scalarization_function import linear_function,cosine_function, utility_function, KL,cauchy_schwarz_function,chebyshev_function
import argparse
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
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

device = torch.device(f"cuda:0" if torch.cuda.is_available() and not False else "cpu")
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_2d(device, hidden_dim, lr, wd, epochs, alpha_r, outdim, criterion,n_tasks,name):
    hnet: nn.Module = Toy_Hypernetwork_2d(ray_hidden_dim = hidden_dim, out_dim = outdim, n_tasks = n_tasks)
    logging.info(f"HN size: {count_parameters(hnet)}")
    hnet = hnet.to(device)
    loss1 = f_1
    loss2 = f_2
    sol = []
    optimizer = torch.optim.Adam(hnet.parameters(), lr = lr, weight_decay=wd)
    start = time.time()
    for epoch in tqdm(range(epochs)):
        ray = torch.from_numpy(
            np.random.dirichlet((alpha_r, alpha_r), 1).astype(np.float32).flatten()
        ).to(device)
        hnet.train()
        optimizer.zero_grad()
        output = hnet(ray)
        ray_cs = 1/ray
        ray = ray.squeeze(0)
        l1 = loss1(output)
        l2 = loss2(output)
        losses = torch.stack((l1, l2)) 
        if criterion == 'ls':
            loss = linear_function(losses,ray)
        elif criterion == 'cheby':
            loss = chebyshev_function(losses,ray)
        elif criterion == 'utility':
            loss = utility_function(losses,ray)
        elif criterion == 'KL':
            loss = KL(losses,ray)
        elif criterion == 'cosine':
            loss = cosine_function(losses,ray)
        elif criterion == 'cauchy':
            loss = cauchy_schwarz_function(losses,ray_cs)
        loss.backward()
        optimizer.step()
        sol.append(output.cpu().detach().numpy().tolist()[0])
    end = time.time()
    time_training = end-start
    torch.save(hnet,("./save_weights/best_weight_"+str(criterion)+"_2d_"+str(name)+".pt"))
    return sol,time_training

def train_3d(device, hidden_dim, lr, wd, epochs,alpha_r, outdim, criterion, n_tasks,name):
    hnet: nn.Module = Toy_Hypernetwork_3d(ray_hidden_dim = hidden_dim, out_dim = outdim, n_tasks = n_tasks)
    logging.info(f"HN size: {count_parameters(hnet)}")
    hnet = hnet.to(device)
    loss1 = f_1
    loss2 = f_2
    loss3 = f_3
    sol = []
    optimizer = torch.optim.Adam(hnet.parameters(), lr = lr, weight_decay=wd)
    start = time.time()
    for epoch in tqdm(range(epochs)):
        ray = torch.from_numpy(
            np.random.dirichlet((alpha_r, alpha_r,alpha_r), 1).astype(np.float32).flatten()
        ).to(device)
        hnet.train()
        optimizer.zero_grad()
        output = hnet(ray)
        output = torch.sqrt(output)
        ray_cs = 1/ray
        ray = ray.squeeze(0)
        l1 = loss1(output)
        l2 = loss2(output)
        l3 = loss3(output)
        losses = torch.stack((l1, l2,l3)) 
        if criterion == 'ls':
            loss = linear_function(losses,ray)
        elif criterion == 'cheby':
            loss = chebyshev_function(losses,ray)
        elif criterion == 'utility':
            loss = utility_function(losses,ray)
        elif criterion == 'KL':
            loss = KL(losses,ray)
        elif criterion == 'cosine':
            loss = cosine_function(losses,ray)
        elif criterion == 'cauchy':
            loss = cauchy_schwarz_function(losses,ray_cs)
        loss.backward()
        optimizer.step()
        sol.append(output.cpu().detach().numpy().tolist()[0])
    end = time.time()
    time_training = end-start
    torch.save(hnet,("./save_weights/best_weight_"+str(criterion)+"_3d_"+str(name)+".pt"))
    return sol,time_training

def draw_2d(sol,pf):
    x = []
    y = []
    for s in sol:
        x.append(f_1(torch.Tensor([s])).item())
        y.append(f_2(torch.Tensor([s])).item())
    plt.scatter(pf[:,0],pf[:,1],s=10,c='gray')
    plt.scatter(x[0],y[0], c = 'y', s = 90,label="Initial Point")
    plt.plot(x[0:2],y[0:2])
    plt.scatter(x[3:],y[3:], c = 'red', s = 20,label="Generated Point")
    plt.title('Pareto front training')
    plt.legend()
    plt.savefig("./train_results/"+str(name)+"_train_"+str(criterion)+".png")
    #plt.savefig("/home/tuantran/Documents/OPT/Multi_Gradient_Descent/PHN/ex2_train_"+str(epochs)+"_ray.pdf")
    plt.show()

def draw_3d(sol,pf):
    x = []
    y = []
    z = []
    for s in sol:
        x.append(f_1(torch.Tensor([s])).item())
        y.append(f_2(torch.Tensor([s])).item())
        z.append(f_3(torch.Tensor([s])).item())
    # pf, pf_tri, ls, tri = create_pf()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_trisurf(pf[:, 0], pf[:, 1], pf[:, 2],
                    color='r', alpha=0.5, shade=True)
    
    graph = ax.scatter(np.array(x), np.array(y), np.array(z), zdir='z',marker='.', s=10, c='black', depthshade=False)
    fake2Dline = mpl.lines.Line2D([0], [0], linestyle="none", c='r',
                                marker='s', alpha=0.5)
    fake2Dline1 = mpl.lines.Line2D([0], [0], linestyle="none", c='black',
                                marker='.', alpha=0.5)
    ax.legend([fake2Dline,fake2Dline1], ['Pareto front','Generated points'], numpoints=1)
    
    ax.set_xlabel(r'$f_1$')
    ax.set_ylabel(r'$f_2$')
    ax.set_zlabel(r'$f_3$')
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_zticks([0.2, 0.4, 0.6, 0.8])
    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.zaxis.set_rotate_label(True)
    ax.xaxis.set_rotate_label(True)
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
    #ax.legend()
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    title = ax.set_title('')
    plt.savefig("./train_results/"+str(name)+"_train_"+str(criterion)+".png")
    def update_graph(i):
        graph._offsets3d = (x[:i+1],y[:i+1],z[:i+1])
        title.set_text('3D Test, iteration = {}'.format(i))
    ani = FuncAnimation(fig, update_graph,len(x), interval=40)
    ax.view_init(5, 45)
    #plt.savefig("./train_results/"+str(name)+"_train_"+str(criterion)+".png")
    #plt.savefig("/home/tuantran/Documents/OPT/Multi_Gradient_Descent/PHN/ex3_train_"+str(epochs)+"_ray.pdf")
    ani.save('./train_results/train.gif', writer='imagemagick', fps=30)
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
    parser.add_argument("--mode", type=str, default='3d', help="mode example")
    parser.add_argument("--name", type=str, default='ex3', help="example name")
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
        sol, time_training = train_2d(device = device, hidden_dim = hidden_dim,
        lr = lr, wd = wd, epochs = epochs, alpha_r = alpha_r, outdim = out_dim,
        criterion = criterion,n_tasks = n_tasks,name = name)
        print("Time: ",time_training)  
        draw_2d(sol,pf)
    else:
        pf  = create_pf_3d()
        sol, time_training = train_3d(device = device, hidden_dim = hidden_dim,
        lr = lr, wd = wd, epochs = epochs, alpha_r = alpha_r, outdim = out_dim,
        criterion = criterion,n_tasks = n_tasks,name = name)
        print("Time: ",time_training)
        draw_3d(sol,pf)