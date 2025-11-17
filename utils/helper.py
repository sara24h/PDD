import torch
import random
import numpy as np
import os


def set_seed(seed):
    """
    Set random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, filename):
    """
    Save model checkpoint
    """
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(filename, model, optimizer=None):
    """
    Load model checkpoint
    """
    if os.path.isfile(filename):
        print(f"Loading checkpoint from {filename}")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        return checkpoint
    else:
        print(f"No checkpoint found at {filename}")
        return None


def count_parameters(model):
    """
    Count total number of parameters in model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer):
    """
    Get current learning rate from optimizer
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
