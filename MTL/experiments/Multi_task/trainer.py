from EPO.epo_train import EPO_train
from LS.ls_train import LS_train
from PMTL.pmtl_train import PMTL_train
from HPN.hpn_train import HPN_train
import torch
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultiTask")
    parser.add_argument(
        "--data-path",
        type=str,
        default="/home/tuantran/Documents/OPT/Multi_Gradient_Descent/HPN-CSF/MTL/dataset/Multi_task",
        help="path to data",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="batch size")
    parser.add_argument(
    "--out-dir",
    type=str,
    default="./outputs",
    help="path to output",
)
    parser.add_argument(
        "--solver", type=str, choices=["ls", "epo","cheby","utility"], default="ls", help="HPN solver"
    )
    parser.add_argument(
        "--method", type=str, choices=["HPN", "LS","PMTL","EPO"], required=True, help="method"
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = args.data_path
    method = args.method
    out_results = args.out_dir
    hpn_solver = args.solver
    batch_size = args.batch_size
    if method == 'HPN':
        HPN_train(device,data_path,out_results,hpn_solver,batch_size)
    elif method == 'LS':
        LS_train(device,data_path,out_results,batch_size)
    elif method == 'PMTL':
        PMTL_train(device,data_path,out_results,batch_size)
    elif method == 'EPO':
        EPO_train(device,data_path,out_results,batch_size)
