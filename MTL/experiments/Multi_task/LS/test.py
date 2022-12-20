import numpy as np
import torch
from matplotlib import pyplot as plt
from pymoo.factory import get_performance_indicator
import os
from data import Dataset
from model_lenet import RegressionModel, RegressionTrain
def hypervolumn(A, ref=None, type='acc'):
    """
    :param A: np.array, num_points, num_task
    :param ref: num_task
    """
    dim = A.shape[1]
    if type == 'loss':
        if ref is None:
            ref = np.ones(dim)
        hv = get_performance_indicator("hv",ref_point=ref)
        return hv.do(A)
    else:
        print('type not implemented')
        return None

def eval_hv(n,dataset,save_weights,test_loader,device):
    init_weight = np.array([0.5 , 0.5 ])
    n_tasks = 2
    model = RegressionTrain(RegressionModel(n_tasks),init_weight)
    model.model.load_state_dict(torch.load(os.path.join(save_weights,'ls_'+str(dataset)+'_lenet_niter_150_'+str(n)+'.pickle')))
    #model.model.load_state_dict(torch.load('/home/tuantran/multiMNIST/saved_model/PMTL_'+str(dataset)+'_lenet_niter_150_npref_5_prefidx_'+str(n)+'.pickle'))
    model = model.to(device)
    model.eval()
    loss_all = []
    num = 0
    losses = np.zeros(2)
    for (it, batch) in enumerate(test_loader):
        
        X = batch[0]
        ts = batch[1]
        bs = len(ts)
        num += bs
        if torch.cuda.is_available():
            X = X.to(device)
            ts = ts.to(device)
        # obtain and store the gradient 
        grads = {}
        losses_vec = []
        
        for i in range(n_tasks):
            #optimizer.zero_grad()
            task_loss = model(X, ts) 
            #print(task_loss.shape)
            losses_vec.append(task_loss[i].detach().cpu().tolist())
        #print(losses_vec)
        losses += bs*np.array(losses_vec)
        #loss_all.append(losses_vec)
    return losses / num
def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = 1e-6 if min_angle is None else min_angle
    ang1 = np.pi / 2 - ang0 if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K, endpoint=True)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y]
def ls_test(dataset,test_loader,save_weights,device):
    loss_hv = []
    for i in range(5):
        tmp = eval_hv(i,dataset,save_weights,test_loader,device)
        loss_hv.append(tmp)
    loss_hv = np.array(loss_hv)
    
    '''
        Multi Mnist
        x = [0.2,0.3,0.4,0.5]
        y = [0.2,0.4,0.6,0.8]
        plt.plot([0.2631959812361983,0.2631959812361983],[0.2,0.8],ls='-.',c='black',label='Single task')
        plt.plot([0.2,0.5],[0.33708666510219815,0.33708666510219815],ls='-.',c='black')
    '''
    '''
        Multi Fashion
        x = [0.4,0.6,0.8,1.0]
        y = [0.4,0.6,0.8,1.0]
        plt.plot([0.4857283249686036,0.4857283249686036],[0.4,1],ls='-.',c='black',label='Single task')
        plt.plot([0.4,1],[0.5331778043433081,0.5331778043433081],ls='-.',c='black')
    '''
    '''
        Multi Fashion Mnist
        x = [0.1,0.4,0.7,1.0]
        y = [0.4,0.6,0.8,1.0]
        plt.plot([0.16867540993645222,0.16867540993645222],[0.4,1],ls='-.',c='black',label = 'Single-task')
        plt.plot([0.1,1],[0.44227917699874203,0.44227917699874203],ls='-.',c='black')
    '''
    '''
        # Accuracy MNIST
        x = [0.82,0.84,0.88,0.92]
        y = [0.78,0.82,0.86,0.9]
        plt.plot([0.91,0.91],[0.78,0.9],ls='-.',label = 'Single-task',c='black')
        plt.plot([0.82,0.92],[0.885,0.885],ls='-.',c='black')   
    '''
    '''
        # Accuracy Fashion
        x = [0.68,0.73,0.78,0.83]
        y = [0.6,0.68,0.76,0.84]
        plt.plot([0.82,0.82],[0.6,0.84],ls='-.',label = 'Single-task',c='black')
        plt.plot([0.68,0.83],[0.80,0.80],ls='-.',c='black')
    '''
    '''
        # Accuracy Fashion + MNIST
        x = [0.6,0.72,0.84,0.96]
        y = [0.59,0.68,0.77,0.86]
        plt.plot([0.94,0.94],[0.59,0.86],ls='-.',label = 'Single-task',c='black')
        plt.plot([0.6,0.96],[0.84,0.84],ls='-.',c='black')
    '''

    if dataset == 'mnist':
        plt.plot(loss_hv[:, 0], loss_hv[:, 1],label = 'EPO',marker='*',linestyle = '-')
        x = [0.2,0.3,0.4,0.5]
        y = [0.2,0.4,0.6,0.8]
        plt.plot([0.2631959812361983,0.2631959812361983],[0.2,0.8],ls='-.',c='black',label='Single task')
        plt.plot([0.2,0.5],[0.33708666510219815,0.33708666510219815],ls='-.',c='black')
        plt.xlabel("Loss CE task left")
        plt.ylabel("Loss CE task right")
        plt.xticks(x)
        plt.yticks(y)
        plt.legend()
        plt.savefig('test_multi_'+str(dataset)+'.jpg')
        plt.savefig('test_multi_'+str(dataset)+'.pdf')
    elif dataset == 'fashion':
        loss_hv = np.array(loss_hv)
        plt.plot(loss_hv[:, 0], loss_hv[:, 1],label = 'EPO',marker='*',linestyle = '-')
        x = [0.4,0.6,0.8,1.0]
        y = [0.4,0.6,0.8,1.0]
        plt.plot([0.4857283249686036,0.4857283249686036],[0.4,1],ls='-.',c='black',label='Single task')
        plt.plot([0.4,1],[0.5331778043433081,0.5331778043433081],ls='-.',c='black')
        plt.xlabel("Loss CE task left")
        plt.ylabel("Loss CE task right")
        plt.xticks(x)
        plt.yticks(y)
        plt.legend()
        plt.savefig('test_multi_'+str(dataset)+'.jpg')
        plt.savefig('test_multi_'+str(dataset)+'.pdf')
    elif dataset == 'fashion_mnist':
        loss_hv = np.array(loss_hv)
        plt.plot(loss_hv[:, 0], loss_hv[:, 1],label = 'EPO',marker='*',linestyle = '-')
        x = [0.1,0.4,0.7,1.0]
        y = [0.4,0.6,0.8,1.0]
        plt.plot([0.16867540993645222,0.16867540993645222],[0.4,1],ls='-.',c='black',label = 'Single-task')
        plt.plot([0.1,1],[0.44227917699874203,0.44227917699874203],ls='-.',c='black')
        plt.xlabel("Loss CE task left")
        plt.ylabel("Loss CE task right")
        plt.xticks(x)
        plt.yticks(y)
        plt.legend()
        plt.savefig('test_multi_'+str(dataset)+'.jpg')
        plt.savefig('test_multi_'+str(dataset)+'.pdf')
    print("HV EPO: ",hypervolumn(np.array(loss_hv), type='loss', ref=np.ones(2) * 2))
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
if __name__ == "__main__":
    dataset = 'mnist'
    test_loader = '/home/tuantran/Documents/OPT/Multi_Gradient_Descent/HPN-CSF/MTL/dataset/Multi_task'
    save_weights = '/home/tuantran/Documents/OPT/Multi_Gradient_Descent/HPN-CSF/MTL/save_weights'
    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = '/home/tuantran/Documents/OPT/Multi_Gradient_Descent/HPN-CSF/MTL/dataset/Multi_task'
    batch_size = 256
    train_loader, val_loader,test_loader = load_data(dataset,data_path,batch_size)
    ls_test(dataset,test_loader,save_weights,device)
