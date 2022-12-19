import numpy as np
import os
import torch
import torch.utils.data
from torch.autograd import Variable
from PMTL.model_lenet import RegressionModel, RegressionTrain
import time
from tqdm import tqdm
from PMTL.data import Dataset
from PMTL.min_norm_solvers import MinNormSolver
def get_d_paretomtl_init(grads,value,weights,i):
    """ 
    calculate the gradient direction for ParetoMTL initialization 
    """
    
    flag = False
    nobj = value.shape
   
    # check active constraints
    current_weight = weights[i]
    rest_weights = weights
    w = rest_weights - current_weight
    
    gx =  torch.matmul(w,value/torch.norm(value))
    idx = gx >  0
   
    # calculate the descent direction
    if torch.sum(idx) <= 0:
        flag = True
        return flag, torch.zeros(nobj)
    if torch.sum(idx) == 1:
        sol = torch.ones(1).cuda().float()
    else:
        vec =  torch.matmul(w[idx],grads)
        sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])


    weight0 =  torch.sum(torch.stack([sol[j] * w[idx][j ,0] for j in torch.arange(0, torch.sum(idx))]))
    weight1 =  torch.sum(torch.stack([sol[j] * w[idx][j ,1] for j in torch.arange(0, torch.sum(idx))]))
    weight = torch.stack([weight0,weight1])
   
    
    return flag, weight


def get_d_paretomtl(grads,value,weights,i):
    """ calculate the gradient direction for ParetoMTL """
    
    # check active constraints
    current_weight = weights[i]
    rest_weights = weights 
    w = rest_weights - current_weight
    
    gx =  torch.matmul(w,value/torch.norm(value))
    idx = gx >  0
    

    # calculate the descent direction
    if torch.sum(idx) <= 0:
        sol, nd = MinNormSolver.find_min_norm_element([[grads[t]] for t in range(len(grads))])
        return torch.tensor(sol).to(device).float()


    vec =  torch.cat((grads, torch.matmul(w[idx],grads)))
    sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])


    weight0 =  sol[0] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,0] for j in torch.arange(2, 2 + torch.sum(idx))]))
    weight1 =  sol[1] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,1] for j in torch.arange(2, 2 + torch.sum(idx))]))
    weight = torch.stack([weight0,weight1])
    
    return weight
def getNumParams(params):
    numParams, numTrainable = 0, 0
    for param in params:
        npParamCount = np.prod(param.data.shape)
        numParams += npParamCount
        if param.requires_grad:
            numTrainable += npParamCount
    return numParams, numTrainable

def train(model,optimizer, niter, preference,pref_idx,train_loader,n_tasks,device):
    print("Preference Vector = {}".format(preference))
    _, n_params = getNumParams(model.parameters())
    print(f"# params={n_params}")
    ref_vec = torch.tensor(np.array(preference)).to(device).float()
    # TRAIN
    # -----
    for t in range(2):
      
        model.train()
        for (it, batch) in enumerate(train_loader):
            X = batch[0]
            ts = batch[1]
            if torch.cuda.is_available():
                X = X.to(device)
                ts = ts.to(device)

            grads = {}
            losses_vec = []
            
            
            # obtain and store the gradient value
            for i in range(n_tasks):
                optimizer.zero_grad()
                task_loss = model(X, ts) 
                losses_vec.append(task_loss[i].data)
                
                task_loss[i].backward()
                
                grads[i] = []
                
                # can use scalable method proposed in the MOO-MTL paper for large scale problem
                # but we keep use the gradient of all parameters in this experiment
                for param in model.parameters():
                    if param.grad is not None:
                        grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))

                
            
            grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
            grads = torch.stack(grads_list)
            
            # calculate the weights
            losses_vec = torch.stack(losses_vec)
            flag, weight_vec = get_d_paretomtl_init(grads,losses_vec,ref_vec,pref_idx)
            
            # early stop once a feasible solution is obtained
            if flag == True:
                print("fealsible solution is obtained.")
                break
            
            # optimization step
            optimizer.zero_grad()
            for i in range(len(task_loss)):
                task_loss = model(X, ts)
                if i == 0:
                    loss_total = weight_vec[i] * task_loss[i]
                else:
                    loss_total = loss_total + weight_vec[i] * task_loss[i]
            
            loss_total.backward()
            optimizer.step()
                
        else:
        # continue if no feasible solution is found
            continue
        # break the loop once a feasible solutions is found
        break
                
        

    # run niter epochs of ParetoMTL 
    for t in tqdm(range(niter)):
        model.train()
        for (it, batch) in enumerate(train_loader):
            
            X = batch[0]
            ts = batch[1]
            if torch.cuda.is_available():
                X = X.to(device)
                ts = ts.to(device)

            # obtain and store the gradient 
            grads = {}
            losses_vec = []
            
            for i in range(n_tasks):
                optimizer.zero_grad()
                task_loss = model(X, ts) 
                losses_vec.append(task_loss[i].data)
                
                task_loss[i].backward()
            
                # can use scalable method proposed in the MOO-MTL paper for large scale problem
                # but we keep use the gradient of all parameters in this experiment              
                grads[i] = []
                for param in model.parameters():
                    if param.grad is not None:
                        grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))

                
                
            grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
            grads = torch.stack(grads_list)
            
            # calculate the weights
            losses_vec = torch.stack(losses_vec)
            weight_vec = get_d_paretomtl(grads,losses_vec,ref_vec,pref_idx)
            
            normalize_coeff = n_tasks / torch.sum(torch.abs(weight_vec))
            weight_vec = weight_vec * normalize_coeff
            
            # optimization step
            optimizer.zero_grad()
            for i in range(len(task_loss)):
                task_loss = model(X, ts)
                if i == 0:
                    loss_total = weight_vec[i] * task_loss[i]
                else:
                    loss_total = loss_total + weight_vec[i] * task_loss[i]
            
            loss_total.backward()
            optimizer.step()
    return model


def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = np.pi / 20. if min_angle is None else min_angle
    ang1 = np.pi * 9 / 20. if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y]


def run(dataset='mnist', base_model='lenet', niter=150,train_loader=None,preferences=None,device = None,out_results=None):
    """
    run Pareto MTL
    """
    
    for i, pref in enumerate(preferences):
        # DEFINE MODEL
        # ---------------------
        if base_model == 'lenet':
            model = RegressionTrain(RegressionModel(2), pref)
            model.to(device)
        # ---------***---------
        # DEFINE OPTIMIZERS
        # -----------------
        # Choose different optimizers for different base model
        n_tasks = 2
        if base_model == 'lenet':
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.)
        model = train(model, optimizer, niter,preferences,i,train_loader,n_tasks,device)
        torch.save(model.model.state_dict(),os.path.join(out_results,'pmtl_'+str(dataset)+'_'+str(base_model)+'_niter_'+str(niter)+'_'+str(i)+'.pickle'))

def load_data(dataset,data_path,batch_size):
    # LOAD DATASET
    # ------------
    # MultiMNIST: multi_mnist.pickle
    if dataset == 'mnist':
        path = os.path.join(data_path,'multi_mnist.pickle')

    # MultiFashionMNIST: multi_fashion.pickle
    if dataset == 'fashion':
        path = os.path.join(data_path,'multi_fashion.pickle')

    # Multi-(Fashion+MNIST): multi_fashion_and_mnist.pickle
    if dataset == 'fashion_and_mnist':
        path = os.path.join(data_path,'multi_fashion_and_mnist.pickle')

    data = Dataset(path, val_size=0.1)
    train_set, val_set, test_set = data.get_datasets()
    batch_size = batch_size
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,num_workers=4,
        shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=batch_size,num_workers=4,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,num_workers=4,
        shuffle=False)
    return train_loader,val_loader, test_loader
def PMTL_train(device,data_path,out_results,batch_size):
    datasets = ['mnist','fashion','fashion_and_mnist']
    for dataset in datasets:
        print("Dataset: ",dataset)
        start = time.time()
        train_loader, val_loader,test_loader = load_data(dataset,data_path,batch_size)
        preferences = np.array([[0.01,0.99],[0.25,0.75],[0.5,0.5],[0.75,0.25],[0.99,0.01]])
        run(dataset=dataset, base_model='lenet', niter=150,train_loader=train_loader,preferences=preferences,device = device,out_results=out_results)
        end = time.time()
        print("Runtime training: ",end-start)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data_path = '/home/tuantran/Documents/OPT/Multi_Gradient_Descent/HPN-CSF/MTL/dataset/Multi_task'
# out_results = '/home/tuantran/Documents/OPT/Multi_Gradient_Descent/HPN-CSF/MTL/experiments/Multi_task/PMTL/outputs'
# PMTL_train(device,data_path,out_results)
