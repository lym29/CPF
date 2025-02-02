import argparse
import os

import torch

_parser = argparse.ArgumentParser(description="MR. Anderson")

_parser.add_argument("--vis_toc", type=float, default=5)
"----------------------------- Experiment options -----------------------------"
_parser.add_argument("-c", "--cfg", help="experiment configure file name", type=str, default=None)
_parser.add_argument("--exp_id", default="default", type=str, help="Experiment ID")

_parser.add_argument("--resume", help="resume training from exp", type=str, default=None)
_parser.add_argument("--resume_epoch", help="resume from the given epoch", type=int, default=0)
_parser.add_argument("--reload", help="reload checkpoint for test", type=str, default=None)

_parser.add_argument("-w", "--workers", help="worker number from data loader", type=int, default=8)
_parser.add_argument("-b",
                     "--batch_size",
                     help="batch size of exp, will replace bs in cfg file if is given",
                     type=int,
                     default=None)

_parser.add_argument("--val_batch_size",
                     help="batch size when val or test, will replace bs in cfg file if is given",
                     type=int,
                     default=1)

_parser.add_argument("--evaluate", help="evaluate the network (ignore training)", action="store_true")
"----------------------------- General options -----------------------------"
_parser.add_argument("-g", "--gpu_id", type=str, default=None, help="override enviroment var CUDA_VISIBLE_DEVICES")
_parser.add_argument("--snapshot", default=5, type=int, help="How often to take a snapshot of the model (0 = never)")
_parser.add_argument("--eval_freq", default=5, type=int, help="How often to evaluate the model on val set")
_parser.add_argument("--log_freq", default=10, type=int, help="How often to write summary logs")
_parser.add_argument("--test_freq", type=int, default=200, help="How often to test, 1 for always -1 for never")
"----------------------------- Distributed options -----------------------------"
_parser.add_argument("--dist_master_addr", type=str, default="localhost")
_parser.add_argument("-p", "--dist_master_port", type=str, default="60000")
_parser.add_argument("--node_rank", default=0, type=int, help="node rank for distributed training")
_parser.add_argument("--nodes", default=1, type=int, help="nodes number for distributed training")
_parser.add_argument("--dist_backend", default="gloo", type=str, help="distributed backend")
_parser.add_argument("-ddp", "--distributed", default=True, type=bool, help="Use distributed data parallel")
"-------------------------------------dataset submit options-------------------------------------"

_parser.add_argument("--submit_dataset", type=str, default="HO3D", choices=["HO3D"])
_parser.add_argument("--callback", type=str, default="dump", choices=["dump", "draw"])


def parse_exp_args():
    arg, custom_arg_string = _parser.parse_known_args()
    return arg, custom_arg_string


def merge_args(arg, cus_arg):
    for k, v in cus_arg.__dict__.items():
        if v is not None:
            arg.__dict__[k] = v
    return arg
