import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from monai.optimizers import LearningRateFinder
import torch.optim as optim
from pathlib import Path

# def save_model(cfg, model, epoch, val_loss,best_loss):


#     if val_loss < best_loss:
        
#         best_loss = val_loss
#         torch.save({
#             'epoch': epoch + 1,
#             'model': model.state_dict(),
#         }, cfg.save_model_dir + f'/model_best.pth')
#         print(f'saved best model in epoch: {epoch+1}')
    
#     elif (epoch + 1) % 5 == 0 or (epoch + 1) == cfg.num_epochs:
        
#         torch.save({
#             'epoch': epoch + 1,
#             'model': model.state_dict(),
#         }, cfg.save_model_dir + f'/model_{epoch + 1}.pth')
#         print(f'saved model in epoch: {epoch+1}')


        
#     return(best_loss)
        
def save_model(cfg, model, epoch, val_loss, best_loss):
    save_dir = Path(cfg.save_model_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    current_saved_path = None

    # 1. Salva se for o melhor modelo (sobrescreve model_best.pth automaticamente)
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
        }, save_dir / 'model_best.pth')
        print(f'saved best model in epoch: {epoch + 1}')
    
    # 2. Salva o checkpoint periódico / final
    # Obs: trocado 'elif' por 'if' para não pular o salvamento periódico se a loss melhorar
    if (epoch + 1) % 5 == 0 or (epoch + 1) == cfg.num_epochs:
        current_saved_path = save_dir / f'model_{epoch + 1}.pth'
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
        }, current_saved_path)
        print(f'saved model in epoch: {epoch + 1}')

        # 3. Deleta versões anteriores da época, preservando o model_best.pth e o atual
        for file in save_dir.glob("model_*.pth"):
            if file.name == "model_best.pth" or file == current_saved_path:
                continue
            try:
                file.unlink()
                print(f'deleted previous checkpoint: {file.name}')
            except OSError as e:
                print(f'Erro ao deletar {file.name}: {e}')

    return best_loss




def get_optimizer(cfg, model, lr):
    if model == 'mai_lab':
        optimizer =   optim.Adam([{'params': encoder.parameters()},
                               {'params': decoder.parameters()}], lr=1e-4, weight_decay=1e-5)
    
    return optimizer