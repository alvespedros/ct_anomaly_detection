import json
from easydict import EasyDict as edict
from ae_training import train
from monai.utils import set_determinism
import torch
import os

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

if __name__ == '__main__':
    set_determinism(seed=42)
    device_ids = [0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device CUDA: ', device)
    

    with open("config/pipeline_test.json", "r") as f:
        cfg = edict(json.load(f))


    # VAE constructor needs image shape
    #im_shape = transforms(train_datadict[0])["im"].shape
    model, avg_train_losses, test_losses = train(cfg,cfg.img_size, cfg.num_epochs, cfg.latent_size, cfg.learning_rate)


