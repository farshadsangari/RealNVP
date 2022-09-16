"""
Utilities of Project
"""

import argparse
import torch
import os
import torch.nn.utils as utils
from yaml import parse
import torch.nn as nn

class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(
        self,
        start_val=0,
        start_count=0,
        start_avg=0,
        start_sum=0
    ):
        self.reset()
        self.val = start_val
        self.avg = start_avg
        self.sum = start_sum
        self.count = start_count

    def reset(self):
        """
            Initialize 'value', 'sum', 'count', and 'avg' with 0.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        """
            Update 'value', 'sum', 'count', and 'avg'.
        """
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def get_args():
    """
        The argument that we have defined, will be used in training and evaluation(infrence) modes
    """
    parser = argparse.ArgumentParser(
        description='Arguemnt Parser of `Train` and `Evaluation` of RealNVP network')

    parser.add_argument('--num-scales', dest='num_scales', default=3,
                        type=int, help='Number of scales in Multiscale structure(Multiscale structure is available in paper)')
    
    parser.add_argument('--num-input-channels', dest='input_channels', default=3,
                        type=int, help='Number of input data channels')
    
    parser.add_argument('--num-middle-channels', dest='middle_channels', default=64,
                        type=int, help='Number of s + t channels')

    parser.add_argument('--batch-size', dest='batch_size', default=64,
                        type=int, help='Number of data in each batch')
    
    parser.add_argument('--lr', dest='lr', default=1e-3,
                        type=float, help='Learning rate value')
    
    parser.add_argument('--weight-decay', dest='weight_decay', default=5e-5,
                        type=float, help='Weight decay value')
    
    parser.add_argument('--num-epochs', dest='num_epochs', default=100,
                        type=float, help='Number of epochs')
    
    parser.add_argument('--gpu', dest='gpu', default=True,
                        type=bool, help='wheather to use gpu or not')
    
    parser.add_argument('--max-grad-norm', dest='max_grad_norm', default=100,
                        type=int, help='Maximum norm of gradients for clipping')
    
    parser.add_argument('--ckpt-save-path', dest='ckpt_save_path', default='../ckpts',
                        type=str, help='base path(folder) to save model ckpts')
    
    parser.add_argument('--ckpt-prefix', dest='ckpt_prefix', default='cktp_epoch_',
                        type=str, help='prefix name of ckpt which you want to save')
    
    parser.add_argument('--ckpt-save-freq', dest='ckpt_save_freq', default=10,
                        type=int, help='after how many epoch(s) save model')
    
    parser.add_argument('--distribution-mean', dest='distribution_mean', default=0,
                        type=float, help='Mean of gussian distribution which we use in loss term')
    
    parser.add_argument('--distribution-std', dest='distribution_std', default=1,
                        type=float, help='Standard deviation of gussian distribution which we use in loss term')
    
    
    parser.add_argument("--ckpt-load-path", dest="ckpt_load_path", type=str, default=None,
                        help="Checkpoints address for loading")
    
    
    parser.add_argument("--possible-values-in-each-input-dimension", dest="possible_values_in_each_input_dimension", type=int, default=256,
                        help="Number of discrete values in each input dimension")
    
    parser.add_argument("--data-constraint", dest="data_constraint", type=float, default=0.9,
                        help="Data constraint which have been used in preprocess step")
    
    parser.add_argument("--report-path", dest="report_path", type=str, default="../reports",
                        help="Saving report directory")
    
    parser.add_argument("--base-raw-data-path", dest="base_raw_data_path", type=str, default="./data/raw_data/img_align_celeba/img_align_celeba",
                        help="base directory of raw datas(images)")
    
    # parser.add_argument("--sample-mode", dest="sample_mode", type=int, default=0,
    #                     help="if sample_mode == 1 , then the model will generate samples")
    
    parser.add_argument("--use-resnet", dest="use_resnet", type=bool, default=True,
                        help="If use_resnet == True , then the residual model will be resnet, otherwise it will be res_unet")
    
    
    parser.add_argument("--num-blocks", dest="num_blocks", type=int, default=2,
                        help="Number of blocks in resnet structure(based on the paper, for CelebA, we could use 2 blocks for residual structure)")
    
    options = parser.parse_args()

    return options


def save_model(file_path, file_name, model, optimizer=None):
    """
        In this function, a model is saved.Usually save model after training in each epoch.
        ------------------------------------------------
        Args:
            - model (torch.nn.Module)
            - optimizer (torch.optim)
            - file_path (str): Path(Folder) for saving the model
            - file_name (str): name of the model checkpoint to save

    """
    state_dict = dict()

    state_dict["model"] = model.state_dict()

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    torch.save(state_dict, os.path.join(file_path, file_name))
    
    
def load_model(ckpt_path, model, optimizer=None):
    """
        Loading a saved model and optimizer (from checkpoint)
    """
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model"])
    if ((optimizer != None) & ("optimizer" in checkpoint.keys())):
        optimizer.load_state_dict(checkpoint["optimizer"])
        
    return model, optimizer


def clip_grad_norm(optimizer, max_norm, norm_type=2):
    for group in optimizer.param_groups:
        utils.clip_grad_norm_(group['params'], max_norm, norm_type)
        
    



