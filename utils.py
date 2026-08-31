import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from monai.optimizers import LearningRateFinder
import torch.optim as optim


def save_model(cfg, model, epoch, val_loss,best_loss):


    if val_loss < best_loss:
        
        best_loss = val_loss
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
        }, cfg.save_model_dir + f'/model_best.pth')
        print(f'saved best model in epoch: {epoch+1}')
    
    elif (epoch + 1) % 5 == 0 or (epoch + 1) == cfg.num_epochs:
        
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
        }, cfg.save_model_dir + f'/model_{epoch + 1}.pth')
        print(f'saved model in epoch: {epoch+1}')


        
    return(best_loss)
        





def get_optimizer(cfg, model, lr):
    if model == 'mai_lab':
        optimizer =   optim.Adam([{'params': encoder.parameters()},
                               {'params': decoder.parameters()}], lr=1e-4, weight_decay=1e-5)
    
    return optimizer