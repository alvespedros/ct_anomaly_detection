import json
from easydict import EasyDict as edict
from train_no_sw import train_model
from test import test_model
from monai.utils import set_determinism
import torch



# if __name__ == '__main__':
#     set_determinism(seed=42)
#     device_ids = [0]
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     with open("cfg_test_pipeline.json", "r") as f:
#         cfg = edict(json.load(f))

    

#     train_model(cfg, device_ids)


if __name__ == '__main__':
    set_determinism(seed=42)
    device_ids = [0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device CUDA: ', device)
    

    with open("config/pipeline_test.json", "r") as f:
        cfg = edict(json.load(f))

    train_model(cfg,device_ids)
