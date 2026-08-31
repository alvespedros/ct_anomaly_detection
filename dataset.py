from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    EnsureTyped,
    RandSpatialCropd,
    SpatialPadd,
    Spacingd,
    ToDeviced,
    Orientationd,
    DeleteItemsd,
    RandRotate90d,
    EnsureTyped,
    ScaleIntensityd,
)
import pandas as pd
#from monai.data import ITKReader
import torch
#from preprocess import get_image
from monai.data import DataLoader, PersistentDataset, Dataset
from monai.utils import set_determinism
import random



def get_dataloader(cfg, mode = 'train'):

    if mode == 'train':


        set_determinism(seed=42)


        # ========== MONAI Dataloader =========
        #IXIT2_filenames = glob(cfg.train_csv + "/*.nii.gz")
        #random.shuffle(IXIT2_filenames)
        df_train = pd.read_csv(cfg.train_csv)
        df_val = pd.read_csv(cfg.valid_csv)
        path_train = df_train['individual_path']
        path_val = df_val['individual_path']


        # Remover duplicatas e estruturar como lista de dicionários para o MONAI
        unique_paths = sorted(list(set(path_train)))
        random.shuffle(unique_paths)
        train_datadict = [{"im": img_path} for img_path in unique_paths][:cfg.train_size]

        unique_paths = sorted(list(set(path_val)))
        val_datadict = [{"im": img_path} for img_path in unique_paths][:cfg.val_size]
        
        #random.shuffle(filepaths)
        

        print(f"total number of train slices: {len(train_datadict)}")
        print(f"total number of val slices: {len(val_datadict)}")



        transforms = Compose(
            [
                # Lê a imagem 2D a partir do caminho
                LoadImaged(keys=["im"], image_only=True),
                
                # Garante a dimensão de canal no início: (H, W) -> (1, H, W)
                # Necessário pois o AutoEncoder espera in_channels=1
                EnsureChannelFirstd(keys=["im"]),


                # Normaliza a intensidade para o intervalo [0, 1]
                ScaleIntensityd(keys=["im"]),
    
                

                # Converte para torch.Tensor com tipo float32 e dispositivo correto
                EnsureTyped(keys=["im"], dtype=torch.float32),
            ]
        )


        rand_transforms = Compose([
                RandRotate90d(
                        keys=["im"],
                        prob=0.5,
                        max_k=3,                 # Número máximo de giros de 90° (1 a 3)
                        spatial_axes=(0, 1),     # Eixos espaciais 2D correspondentes a (H, W)
)

        ])


        train_transforms = Compose(transforms.transforms + rand_transforms.transforms)
        #train_transforms = Compose(deterministic_transforms.transforms)
        #train_ds = Dataset(train_datadict, train_transforms)
        train_ds = PersistentDataset(train_datadict, train_transforms, cache_dir = cfg.cache_dir)
        
        train_loader = DataLoader(train_ds, batch_size=cfg.train_batch_size, num_workers=cfg.num_workers, shuffle=True, prefetch_factor = cfg.prefetch_factor)
        
        #val_ds = Dataset(val_datadict, transforms)
        val_ds = PersistentDataset(val_datadict, transforms, cache_dir = cfg.cache_dir)
        val_loader = DataLoader(val_ds, batch_size=cfg.val_batch_size, num_workers=cfg.num_workers, shuffle=False,prefetch_factor = cfg.prefetch_factor)
        
        return(train_loader, val_loader)
    
    elif mode == 'test':

        # load dataset
        df = pd.read_csv(cfg.train_csv)
        df = df[df['Normal'] == 0]
        df  = df.sample(n=1)
        filepaths = [{"im": fname} for fname in df['Path']]
        filepaths = [{"im": p['im'].replace('D:/Backup/Mestrado/dados/', '/home/jovyan/imgs/').replace('\\', '/')} for p in filepaths]
        filepaths = [{"im": p['im'].replace('D:/Mestrado/dados/', '/home/jovyan/imgs/').replace('\\', '/')} for p in filepaths]
        
        random.shuffle(filepaths)
        

        deterministic_transforms = Compose(
    [
        LoadImaged(keys=["im"], image_only=False, reader=ITKReader(series_name="")),
        EnsureTyped(keys=["im"]),
        EnsureChannelFirstd(keys=["im"]),
        ToDeviced(keys=["im"], device = cfg.device),
        Orientationd(keys=["im"], axcodes="RAS"),
        ScaleIntensityRanged(
        keys=["im"],
        a_min=-15,
        a_max=85,
        b_min=0.0,
        b_max=1.0,
        clip=True
    ),
        Spacingd(
            keys=["im"],
            pixdim=(1.0, 1.0, 1.0),
            mode="bilinear"
        ),
        Resized(keys = ['im'], spatial_size = (160,192,96)),
        DeleteItemsd(keys=["image_meta_dict"]),  # drop all metadata entirely
        
        
    ])
        ds = PersistentDataset(filepaths, deterministic_transforms, cache_dir = cfg.cache_dir)
        test_loader = DataLoader(ds, batch_size=1)

        return(test_loader)
