import sys
import os
sys.path.append(os.getcwd())
import time
from tqdm import tqdm
from models import Hypernetwork
import numpy as np
import torch
from matplotlib import pyplot as plt
from create_pareto_front import PF
from scalarization_function import CS_functions,EPOSolver
import argparse
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from tools.hv import HvMaximization
import yaml
from torch.autograd import Variable

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 18,
         }
font_legend = {'family': 'Times New Roman',
               'weight': 'normal',
                'size': 18,
               }
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def circle_points_random(r, n):
    """
    generate n random unit vectors
    """
    
    circles = []
    for r, n in zip(r, n):
        t = np.random.rand(n) * 0.5 * np.pi  
        t = np.sort(t)
        
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    #print(circles)
    return circles
def train_2d(device, cfg,criterion):
    name = cfg['NAME']
    mode = cfg['MODE']
    ray_hidden_dim = cfg['TRAIN']['Ray_hidden_dim']
    out_dim = cfg['TRAIN']['Out_dim']
    n_tasks = cfg['TRAIN']['N_task']
    num_hidden_layer = cfg['TRAIN']['Solver'][criterion]['Num_hidden_layer']
    last_activation = cfg['TRAIN']['Solver'][criterion]['Last_activation']
    ref_point = tuple(map(int, cfg['TRAIN']['Ref_point'].split(',')))
    lr = cfg['TRAIN']['OPTIMIZER']['Lr']
    wd = cfg['TRAIN']['OPTIMIZER']['WEIGHT_DECAY']
    type_opt = cfg['TRAIN']['OPTIMIZER']['TYPE']
    epochs = cfg['TRAIN']['Epoch']
    alpha_r = cfg['TRAIN']['Alpha']
    
    loss1 = f_1
    loss2 = f_2
    
    start = time.time()
    if criterion == 'HVI':
        
        sol = []
        head = cfg['TRAIN']['Solver'][criterion]['Head']
        partition = np.array([1/head]*head)
        hnet = Hypernetwork(ray_hidden_dim = ray_hidden_dim, out_dim = out_dim, n_tasks = n_tasks,num_hidden_layer=num_hidden_layer,last_activation=last_activation)
        hnet = hnet.to(device)
        if type_opt == 'adam':
            optimizer = torch.optim.Adam(hnet.parameters(), lr = lr, weight_decay=wd)
        net_list = []

        n_mo_sol = cfg['TRAIN']['Solver'][criterion]['Head']
        n_mo_obj = cfg['TRAIN']['N_task']
        hesophat = cfg['TRAIN']['Solver'][criterion]['Rho']
        start = 0.
        end = np.pi/2
        print("Start Phase 1")
        #phase1_iter = trange(150)

        optimizer_direction = torch.optim.Adam(hnet.parameters(), lr=5e-4, weight_decay=wd)
        for _ in range(150):
            optimizer_direction.zero_grad()
            if (_+1) % 50 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= np.sqrt(0.5)
                #patience = 0
                lr *= np.sqrt(0.5)    
            penalty_epoch = []

            losses_mean= []
            weights = []
            outputs = []
            rays = []
            penalty = []
            sum_penalty = 0
            for j in range(n_mo_sol):

                random = np.random.uniform(start + j*(end-start)/head, start+ (j+1)*(end-start)/head)
                #random = np.random.uniform(start, end)
                ray = torch.from_numpy(
                np.random.dirichlet((alpha_r, alpha_r), 1).astype(np.float32).flatten()
                ).to(device)
                            
                rays.append(ray)

                outputs.append(hnet(rays[j]))
                #outputs.append(net_list[j](weights[j])[0])
                losses_mean.append(torch.stack([loss1(outputs[j]), loss2(outputs[j])]))

                penalty.append(torch.sum(losses_mean[j]*rays[j])/(torch.norm(rays[j])*torch.norm(losses_mean[j])))

                sum_penalty += penalty[j].item()

            direction_loss = 0.
            for phat in penalty[:]:
                direction_loss -= phat
            direction_loss.backward()
            optimizer_direction.step()

            aaa = sum_penalty/float(head)
            #phase1_iter.set_description(f"Epochs {_} penalty {aaa:.2f}")

        print("End phase 1 after {}".format(_))
        mo_opt = HvMaximization(n_mo_sol, n_mo_obj, ref_point)
        
        dem = 0
        #epoch_iter = trange(epochs)
        for epoch in tqdm(range(epochs)):
            dem += 1

            hnet.train()
            optimizer.zero_grad()

            loss_torch_per_sample = []
            loss_numpy_per_sample = []
            loss_per_sample = []
            weights = []
            outputs = []
            rays = []
            penalty = []
            for i in range(n_mo_sol):
                random = np.random.uniform(start, end)
                #random = np.random.uniform(start + i*(end-start)/head, start+ (i+1)*(end-start)/head)
                ray = torch.from_numpy(
                np.random.dirichlet((alpha_r, alpha_r), 1).astype(np.float32).flatten()
                ).to(device)
                rays.append(ray)
                output = hnet(rays[i])
                
                outputs.append(output)

                #outputs.append(net_list[i](weights[i])[0])
                
                loss_per_sample = torch.stack([loss1(outputs[i]), loss2(outputs[i])])

                loss_torch_per_sample.append(loss_per_sample)
                loss_numpy_per_sample.append(loss_per_sample.cpu().detach().numpy())

                penalty.append(torch.sum(loss_torch_per_sample[i]*rays[i])/
                                (torch.norm(loss_torch_per_sample[i])*torch.norm(rays[i])))

            loss_numpy_per_sample = np.array(loss_numpy_per_sample)[np.newaxis, :, :].transpose(0, 2, 1) #n_samples, obj, sol

            n_samples = 1
            dynamic_weights_per_sample = torch.ones(n_mo_sol, n_mo_obj, n_samples)
            for i_sample in range(0, n_samples):
                weights_task = mo_opt.compute_weights(loss_numpy_per_sample[i_sample,:,:])
                dynamic_weights_per_sample[:, :, i_sample] = weights_task.permute(1,0)
            
            dynamic_weights_per_sample = dynamic_weights_per_sample.to(device)
            i_mo_sol = 0
            total_dynamic_loss = torch.mean(torch.sum(dynamic_weights_per_sample[i_mo_sol, :, :]
                                                    * loss_torch_per_sample[i_mo_sol], dim=0))
            for i_mo_sol in range(1, len(net_list)):
                total_dynamic_loss += torch.mean(torch.sum(dynamic_weights_per_sample[i_mo_sol, :, :] 
                                                * loss_torch_per_sample[i_mo_sol], dim=0))
            
            for idx in range(head):
                total_dynamic_loss -= hesophat*penalty[idx]*partition[idx]
            
            
            total_dynamic_loss /= head
            total_dynamic_loss.backward()
            optimizer.step()

            penalty = [i.item() for i in penalty]
            sol.append(output.cpu().detach().numpy().tolist()[0])
    else:
        hnet = Hypernetwork(ray_hidden_dim = ray_hidden_dim, out_dim = out_dim, n_tasks = n_tasks,num_hidden_layer=num_hidden_layer,last_activation=last_activation)
        hnet = hnet.to(device)
        if type_opt == 'adam':
            optimizer = torch.optim.Adam(hnet.parameters(), lr = lr, weight_decay=wd)
        sol = []
        for epoch in tqdm(range(epochs)):
            ray = torch.from_numpy(
                np.random.dirichlet((alpha_r, alpha_r), 1).astype(np.float32).flatten()
            ).to(device)
            #print(ray)
            hnet.train()
            optimizer.zero_grad()
            output = hnet(ray)
            ray_cs = 1/ray
            ray = ray.squeeze(0)
            l1 = loss1(output)
            l2 = loss2(output)
            losses = torch.stack((l1, l2)) 
            CS_func = CS_functions(losses,ray)

            if criterion == 'Prod':
                loss = CS_func.product_function()
            elif criterion == 'Log':
                loss = CS_func.log_function()
            elif criterion == 'AC':
                rho = cfg['TRAIN']['Solver'][criterion]['Rho']
                loss = CS_func.ac_function(rho = rho)
            elif criterion == 'MC':
                rho = cfg['TRAIN']['Solver'][criterion]['Rho']
                loss = CS_func.mc_function(rho = rho)
            elif criterion == 'HV':
                mo_opt = HvMaximization(1, n_tasks, ref_point)
                loss_numpy = []
                for j in range(1):
                    loss_numpy.append(losses.detach().cpu().numpy())
                loss_numpy = np.array(loss_numpy).T
                loss_numpy = loss_numpy[np.newaxis, :, :]
                rho = cfg['TRAIN']['Solver'][criterion]['Rho']
                dynamic_weight = mo_opt.compute_weights(loss_numpy[0,:,:])
                loss = CS_func.hv_function(dynamic_weight.reshape(1,2),rho =rho)
            elif criterion == 'HVI':
                loss_numpy = []
                n_samples = 3
                for j in range(n_samples):
                    loss_numpy.append(losses.detach().cpu().numpy())
                loss_numpy = np.array(loss_numpy).T
                loss_numpy = loss_numpy[np.newaxis, :, :]
                rho = cfg['TRAIN']['Solver'][criterion]['Rho']
                dynamic_weights_per_sample = torch.ones(1, 2, n_samples)
                for i_sample in range(n_samples):
                    dynamic_weight = mo_opt.compute_weights(loss_numpy[j,:,:])
                    dynamic_weights_per_sample[:, :, i_sample] = dynamic_weight.permute(1,0)
                dynamic_weights_per_sample = dynamic_weights_per_sample.to(device)
                loss = CS_func.hv_function(dynamic_weight.reshape(1,2),rho =rho)
            elif criterion == 'LS':
                loss = CS_func.linear_function()
            elif criterion == 'Cheby':
                loss = CS_func.chebyshev_function()
            elif criterion == 'Utility':
                ub = cfg['TRAIN']['Solver'][criterion]['Ub']
                loss = CS_func.utility_function(ub = ub)
            elif criterion == 'KL':
                loss = CS_func.KL_function()
            elif criterion == 'Cosine':
                loss = CS_func.cosine_function()
            elif criterion == 'Cauchy':
                CS_func = CS_functions(losses,ray_cs) 
                loss = CS_func.cauchy_schwarz_function()
            elif criterion == 'EPO':
                solver = EPOSolver(n_tasks=2, n_params=count_parameters(hnet))
                loss = solver(losses, ray, list(hnet.parameters()))
            elif criterion == 'CPMTL':
                CS_func = CS_functions(losses,ray_cs)
                optimizer.zero_grad()
                l1.backward(retain_graph = True)
                grad_ = {}
                grad_[1] = []
                
                for param in hnet.parameters():
                    if param.grad is not None:
                        grad_[1].append(Variable(param.grad.data.clone().flatten(), requires_grad = False))
                optimizer.zero_grad()
                l2.backward(retain_graph = True)
                grad_[2] = []
                for param in hnet.parameters():
                    if param.grad is not None:
                        grad_[2].append(Variable(param.grad.data.clone().flatten(), requires_grad = False))
                grads_list = [torch.cat(grad_[1]),torch.cat(grad_[2])]
                grads = torch.stack(grads_list)
                context = circle_points_random([1], [25])[0]
                #print(context.shape)
                #context = context/np.sum(context,axis = 1).reshape(25,1)
                ref_vec = torch.tensor(context).float()
                
                #print(ref_vec)
                pref_idx = np.random.randint(25)
                #loss = CS_func.get_d_paretomtl(grads,losses,ref_vec,ref_vec[pref_idx])
                loss = CS_func.get_d_paretomtl(grads,losses,ref_vec,ray)
                #print(loss.item())
            loss.backward()
            optimizer.step()
            if criterion == 'CPMTL':
                r1 = np.random.rand(1)
                r2 = np.random.rand(1)
                if r1 < 0.1:
                    optimizer.zero_grad()
                    ray = torch.tensor([0,1]).float()
                    output = hnet(ray)
                    #ray = ray.squeeze(0)
                    l1 = loss1(output)
                    l2 = loss2(output)
                    losses = torch.stack((l1, l2)) 
                    loss = l1
                    loss.backward()
                    optimizer.step()
                if r2 < 0.1:
                    optimizer.zero_grad()
                    ray = torch.tensor([1,0]).float()
                    output = hnet(ray)
                    #ray = ray.squeeze(0)
                    l1 = loss1(output)
                    l2 = loss2(output)
                    losses = torch.stack((l1, l2)) 
                    loss = l2
                    loss.backward()
                    optimizer.step()
            sol.append(output.cpu().detach().numpy().tolist()[0])
    end = time.time()
    time_training = end-start
    torch.save(hnet,("./save_weights/best_weight_"+str(criterion)+"_"+str(mode)+"_"+str(name)+".pt"))
    return sol,time_training

def train_3d(device, cfg, criterion):
    name = cfg['NAME']
    mode = cfg['MODE']
    ray_hidden_dim = cfg['TRAIN']['Ray_hidden_dim']
    out_dim = cfg['TRAIN']['Out_dim']
    n_tasks = cfg['TRAIN']['N_task']
    num_hidden_layer = cfg['TRAIN']['Solver'][criterion]['Num_hidden_layer']
    last_activation = cfg['TRAIN']['Solver'][criterion]['Last_activation']
    ref_point = tuple(map(int, cfg['TRAIN']['Ref_point'].split(',')))
    lr = cfg['TRAIN']['OPTIMIZER']['Lr']
    wd = cfg['TRAIN']['OPTIMIZER']['WEIGHT_DECAY']
    type_opt = cfg['TRAIN']['OPTIMIZER']['TYPE']
    epochs = cfg['TRAIN']['Epoch']
    alpha_r = cfg['TRAIN']['Alpha']
    hnet = Hypernetwork(ray_hidden_dim = ray_hidden_dim, out_dim = out_dim, n_tasks = n_tasks,num_hidden_layer=num_hidden_layer,last_activation=last_activation)
    hnet = hnet.to(device)
    loss1 = f_1
    loss2 = f_2
    loss3 = f_3
    sol = []
    if type_opt == 'adam':
        optimizer = torch.optim.Adam(hnet.parameters(), lr = lr, weight_decay=wd) 
    mo_opt = HvMaximization(1, n_tasks, ref_point)
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
        CS_func = CS_functions(losses,ray)

        if criterion == 'Prod':
            loss = CS_func.product_function()
        elif criterion == 'Log':
            loss = CS_func.log_function()
        elif criterion == 'AC':
            rho = cfg['TRAIN']['Solver'][criterion]['Rho']
            loss = CS_func.ac_function(rho = rho)
        elif criterion == 'MC':
            rho = cfg['TRAIN']['Solver'][criterion]['Rho']
            loss = CS_func.mc_function(rho = rho)
        elif criterion == 'HV':
            loss_numpy = []
            for j in range(1):
                loss_numpy.append(losses.detach().cpu().numpy())
            loss_numpy = np.array(loss_numpy).T
            loss_numpy = loss_numpy[np.newaxis, :, :]
            rho = cfg['TRAIN']['Solver'][criterion]['Rho']
            dynamic_weight = mo_opt.compute_weights(loss_numpy[0,:,:])
            loss = CS_func.hv_function(dynamic_weight.reshape(1,3),rho = rho)
        elif criterion == 'LS':
            loss = CS_func.linear_function()
        elif criterion == 'Cheby':
            loss = CS_func.chebyshev_function()
        elif criterion == 'Utility':
            ub = cfg['TRAIN']['Solver'][criterion]['Ub']
            loss = CS_func.utility_function(ub = ub)
        elif criterion == 'KL':
            loss = CS_func.KL_function()
        elif criterion == 'Cosine':
            loss = CS_func.cosine_function()
        elif criterion == 'Cauchy':
            CS_func = CS_functions(losses,ray_cs)
            loss = CS_func.cauchy_schwarz_function()
        loss.backward()
        optimizer.step()
        sol.append(output.cpu().detach().numpy().tolist()[0])
    end = time.time()
    time_training = end-start
    #torch.save(hnet,("./save_weights/best_weight_"+str(criterion)+"_"+str(mode)+"_"+str(name)+".pt"))
    return sol,time_training

def draw_2d(sol,pf,cfg,criterion):
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
    plt.savefig("./train_results/"+str(cfg['NAME'])+"_train_"+str(criterion)+".png")
    #plt.savefig("/home/tuantran/Documents/OPT/Multi_Gradient_Descent/PHN/ex2_train_"+str(epochs)+"_ray.pdf")
    plt.show()

def draw_3d(sol,pf,cfg,criterion):
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
    plt.savefig("./train_results/"+str(cfg['NAME'])+"_train_"+str(criterion)+".png")
    def update_graph(i):
        graph._offsets3d = (x[:i+1],y[:i+1],z[:i+1])
        title.set_text('3D Test, iteration = {}'.format(i))
    #ani = FuncAnimation(fig, update_graph,len(x), interval=40)
    ax.view_init(5, 45)
    #plt.savefig("./train_results/"+str(name)+"_train_"+str(criterion)+".png")
    #plt.savefig("/home/tuantran/Documents/OPT/Multi_Gradient_Descent/PHN/ex3_train_"+str(epochs)+"_ray.pdf")
    #ani.save('./train_results/train.gif', writer='imagemagick', fps=30)
    plt.show()

def main(cfg,criterion,device,cpf):
    if cfg['MODE'] == '2d':
        if cfg['NAME'] == 'ex1':
            
            pf = cpf.create_pf5() 
        else:
            pf = cpf.create_pf6() 
        sol, time_training = train_2d(device,cfg,criterion)
        print("Time: ",time_training)  
        draw_2d(sol,pf,cfg,criterion)
    else:
        pf  = cpf.create_pf_3d()
        sol, time_training = train_3d(device,cfg,criterion)
        print("Time: ",time_training)  
        draw_3d(sol,pf,cfg,criterion)

if __name__ == "__main__":
    device = torch.device(f"cuda:0" if torch.cuda.is_available() and not False else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/ex1.yaml', help="config file")
    parser.add_argument(
        "--solver", type=str, choices=["LS", "KL","Cheby","Utility","Cosine","Cauchy","Prod","Log","AC","MC","HV","CPMTL","EPO","HVI"], default="Utility", help="solver"
    )
    args = parser.parse_args()
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
