import logging
import argparse
from collections import defaultdict
import time
import numpy as np
import torch
from torch import nn
from tqdm import trange
import os
from HPN.metrics import hypervolumn
from HPN.data import Dataset
from HPN.models import (
    LeNetHyper,
    LeNetTarget,
    ResnetHyper,
    ResNetTarget,
)
from HPN.utils import (
    circle_points,
    count_parameters,
    get_device,
    save_args,
    set_logger,
    set_seed,
)
from HPN.solver import EPOSolver, LinearScalarizationSolver, ChebyshevBasedSolver, UtilityBasedSolver

@torch.no_grad()
def evaluate_hv(hypernet, targetnet, loader, rays, device):
    hypernet.eval()
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.CrossEntropyLoss()
    # num_samples = 0
    results = defaultdict(list)
    losses_all = []
    
    for ray in rays:
        total = 0.0
        task1_correct, task2_correct = 0.0, 0.0
        l1, l2 = 0.0, 0.0
        ray = torch.from_numpy(ray.astype(np.float32)).to(device)
        ray /= ray.sum()
        num_samples = 0
        losses = np.zeros(2)
        for batch in loader:
            hypernet.zero_grad()

            batch = (t.to(device) for t in batch)
            img, ys = batch
            bs = len(ys)
            num_samples += bs
            weights = hypernet(ray)
            logit1, logit2 = targetnet(img, weights)

            # loss
            curr_l1 = loss1(logit1, ys[:, 0])
            curr_l2 = loss2(logit2, ys[:, 1])
            losses_batch = [curr_l1.detach().cpu().tolist(),curr_l2.detach().cpu().tolist()]
            losses += bs * np.array(losses_batch)
        losses /= num_samples
        losses_all.append(losses)


    return losses_all

def train(
    dataset,
    path,
    solver_type: str,
    epochs: int,
    hidden_dim: int,
    model: str,
    lr: float,
    wd: float,
    bs: int,
    val_size: float,
    n_rays: int,
    alpha: float,
    no_val_eval: bool,
    out_dir: str,
    device: torch.device,
    eval_every: int,
) -> None:
    # ----
    # Nets
    # ----
    if model == "lenet":
        hnet: nn.Module = LeNetHyper([9, 5], ray_hidden_dim=hidden_dim)
        net: nn.Module = LeNetTarget([9, 5])

    logging.info(f"HN size: {count_parameters(hnet)}")

    hnet = hnet.to(device)
    net = net.to(device)

    # ---------
    # Task loss
    # ---------
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(hnet.parameters(), lr=lr, weight_decay=wd)
    if solver_type == "epo":
        solver = EPOSolver(n_tasks=2, n_params=count_parameters(hnet))
    elif solver_type == "ls":
        # ls,cheby,utility
        solver = LinearScalarizationSolver()
    elif solver_type == "cheby":
        # ls,cheby,utility
        solver = ChebyshevBasedSolver(lower_bound = 0.1)
    elif solver_type == "utility":
        # ls,cheby,utility
        solver = UtilityBasedSolver(upper_bound = 200.0)

    # ----
    # data
    # ----
    assert val_size > 0, "please use validation by providing val_size > 0"
    train_loader, val_loader,test_loader = load_data(dataset,path)
    min_angle = 0.1
    max_angle = np.pi / 2 - 0.1
    test_rays = circle_points(n_rays, min_angle=min_angle, max_angle=max_angle)

    # ----------
    # Train loop
    # ----------
    last_eval = -1
    epoch_iter = trange(epochs)

    val_results = dict()
    test_results = dict()
    best_hv_loss = -1
    start = time.time()
    for epoch in epoch_iter:
        losses_epoch = []
        for i, batch in enumerate(train_loader):
            hnet.train()
            optimizer.zero_grad()
            img, ys = batch
            img = img.to(device)
            ys = ys.to(device)

            if alpha > 0:
                ray = torch.from_numpy(
                    np.random.dirichlet((alpha, alpha), 1).astype(np.float32).flatten()
                ).to(device)
            else:
                alpha = torch.empty(
                    1,
                ).uniform_(0.0, 1.0)
                ray = torch.tensor([alpha.item(), 1 - alpha.item()]).to(device)

            weights = hnet(ray)
            logit1, logit2 = net(img, weights)

            l1 = loss1(logit1, ys[:, 0])
            l2 = loss2(logit2, ys[:, 1])
            losses = torch.stack((l1, l2))

            ray = ray.squeeze(0)
            loss = solver(losses, ray, list(hnet.parameters()))
            loss.backward()

            epoch_iter.set_description(
                f"total weighted loss: {loss.item():.3f}, loss 1: {l1.item():.3f}, loss 2: {l2.item():.3f}"
            )

            optimizer.step()
        loss_hv = evaluate_hv(
                    hypernet=hnet,
                    targetnet=net,
                    loader=val_loader,
                    rays=test_rays,
                    device=device,
                )
        hv_loss = hypervolumn(np.array(loss_hv), type='loss', ref=np.ones(2) * 2)
        print("Epoch"+str(epoch)+": ",hv_loss)
        if hv_loss>best_hv_loss:
            best_hv_loss = hv_loss
            print("Update best model")
            torch.save(hnet,os.path.join(out_dir,"best_model_"+str(solver_type)+"_"+str(dataset)+".pt"))

def load_data(dataset,data_path):
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
    batch_size = 256
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
def HPN_train(device,data_path,out_results,solver,batch_size):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--n-epochs", type=int, default=150, help="num. epochs")
    # parser.add_argument(
    #     "--ray-hidden", type=int, default=100, help="lower range for ray"
    # )
    # parser.add_argument("--alpha", type=float, default=0.2, help="alpha for dirichlet")
    # parser.add_argument(
    #     "--model",
    #     type=str,
    #     default="lenet",
    #     choices=["lenet", "resnet"],
    #     help="model name",
    # )

    # parser.add_argument("--gpus", type=str, default="0", help="gpu device")
    # # parser.add_argument("--batch-size", type=int, default=256, help="batch size")
    # parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    # parser.add_argument("--wd", type=float, default=0.0, help="weight decay")
    # parser.add_argument("--val-size", type=float, default=0.1, help="validation size")
    # parser.add_argument(
    #     "--no-val-eval",
    #     action="store_true",
    #     default=False,
    #     help="evaluate on validation",
    # )
    # # parser.add_argument(
    # #     "--solver", type=str, choices=["ls", "epo","cheby","utility"], default="ls", help="solver"
    # # )
    # parser.add_argument(
    #     "--eval-every",
    #     type=int,
    #     default=10,
    #     help="number of epochs between evaluations",
    # )
    # parser.add_argument("--n-rays", type=int, default=25, help="num. rays")
    # parser.add_argument("--seed", type=int, default=42, help="random seed")
    # args = parser.parse_args()

    set_seed(42)
    set_logger()
    datasets = ['mnist','fashion','fashion_and_mnist']
    for dataset in datasets:
        print("Dataset: ",dataset)
        print("Solver: ",solver)
        start = time.time()
        train(
            dataset = dataset,
            path=data_path,
            solver_type=solver,
            epochs=150,
            hidden_dim=100,
            model='lenet',
            lr=1e-4,
            wd=0.0,
            bs=batch_size,
            device=device,
            eval_every=10,
            no_val_eval=False,
            val_size=0.1,
            n_rays=25,
            alpha=0.2,
            out_dir=out_results,
        )
        end = time.time()
        print("Runtime training: ",end-start)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data_path = '/home/tuantran/Documents/OPT/Multi_Gradient_Descent/HPN-CSF/MTL/dataset/Multi_task'
# out_results = '/home/tuantran/Documents/OPT/Multi_Gradient_Descent/HPN-CSF/MTL/experiments/Multi_task/HPN/outputs'
# hpn_solver = 'ls'
# HPN_train(device,data_path,out_results,hpn_solver)

