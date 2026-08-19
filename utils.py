import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from monai.optimizers import LearningRateFinder
import torch.optim as optim





def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD




def get_optimizer(cfg, model, lr):
    if model == 'mai_lab':
        optimizer =   optim.Adam([{'params': encoder.parameters()},
                               {'params': decoder.parameters()}], lr=1e-4, weight_decay=1e-5)
    
    return optimizer