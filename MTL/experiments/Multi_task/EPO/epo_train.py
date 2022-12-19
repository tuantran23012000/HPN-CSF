import numpy as np
import os
import torch
import torch.utils.data
from torch.autograd import Variable
from EPO.model_lenet import RegressionModel, RegressionTrain
from EPO.epo_lp import EPO_LP
import time
from tqdm import tqdm
from EPO.data import Dataset
def getNumParams(params):
    numParams, numTrainable = 0, 0
    for param in params:
        npParamCount = np.prod(param.data.shape)
        numParams += npParamCount
        if param.requires_grad:
            numTrainable += npParamCount
    return numParams, numTrainable

def train(model,optimizer, niter, preference,train_loader,n_tasks,device):
    print("Preference Vector = {}".format(preference))
    # Instantia EPO Linear Program Solver
    _, n_params = getNumParams(model.parameters())
    print(f"# params={n_params}")
    epo_lp = EPO_LP(m=n_tasks, n=n_params, r=preference)

    # TRAIN
    # -----
    for t in tqdm(range(niter)):
        n_manual_adjusts = 0
        descent = 0.
        model.train()
        for (it, batch) in enumerate(train_loader):
            X = batch[0]
            ts = batch[1]
            if torch.cuda.is_available():
                X = X.to(device)
                ts = ts.to(device)
            # Obtain losses and gradients
            grads = {}
            losses = []
            for i in range(n_tasks):
                optimizer.zero_grad()
                task_loss = model(X, ts)
                losses.append(task_loss[i].data.cpu().numpy())
                task_loss[i].backward()

                # One can use scalable method proposed in the MOO-MTL paper 
                # for large scale problem; but we use the gradient
                # of all parameters in this experiment.
                grads[i] = []
                for param in model.parameters():
                    if param.grad is not None:
                        grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))

            grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
            G = torch.stack(grads_list)
            GG = G @ G.T
            losses = np.stack(losses)

            try:
                # Calculate the alphas from the LP solver
                alpha = epo_lp.get_alpha(losses, G=GG.cpu().numpy(), C=True)
                if epo_lp.last_move == "dom":
                    descent += 1
            except Exception as e:
                #print(e)
                alpha = None
            if alpha is None:   # A patch for the issue in cvxpy
                alpha = preference / preference.sum()
                n_manual_adjusts += 1
            alpha = n_tasks * torch.from_numpy(alpha).to(device)
            # Optimization step
            optimizer.zero_grad()
            task_losses = model(X, ts)
            weighted_loss = torch.sum(task_losses * alpha)  # * 5. * max(epo_lp.mu_rl, 0.2)
            weighted_loss.backward()
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


def run(dataset='mnist', base_model='lenet', niter=100,train_loader=None,preferences=None,device = None,out_results=None):
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
        model = train(model, optimizer, niter, pref,train_loader,n_tasks,device)
        torch.save(model.model.state_dict(),os.path.join(out_results,'epo_'+str(dataset)+'_'+str(base_model)+'_niter_'+str(niter)+'_'+str(i)+'.pickle'))

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
def EPO_train(device,data_path,out_results,batch_size):
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
# out_results = '/home/tuantran/Documents/OPT/Multi_Gradient_Descent/HPN-CSF/MTL/experiments/Multi_task/EPO/outputs'
# EPO_train(device,data_path,out_results)

