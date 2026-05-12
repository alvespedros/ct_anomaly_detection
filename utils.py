import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from monai.optimizers import LearningRateFinder
import torch.optim as optim



def get_optimizer(cfg, model, lr):
    if model == 'mai_lab':
        optimizer =   optim.Adam([{'params': encoder.parameters()},
                               {'params': decoder.parameters()}], lr=1e-4, weight_decay=1e-5)
    
    return optimizer