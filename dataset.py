from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    EnsureTyped,
    Resized,
    RandSpatialCropd,
    SpatialPadd,
    Spacingd,
    ToDeviced,
    Orientationd,
    DeleteItemsd,
    SqueezeDimd,
    MaskIntensityd
)
import pandas as pd
from glob import glob
from monai.data import ITKReader

#from preprocess import get_image
from monai.data import DataLoader, PersistentDataset
from monai.utils import set_determinism
import random



def get_dataloader(cfg, mode = 'train'):

    if mode == 'train':


        set_determinism(seed=42)


        # ========== MONAI Dataloader =========
        #IXIT2_filenames = glob(cfg.train_csv + "/*.nii.gz")
        #random.shuffle(IXIT2_filenames)
        df = pd.read_csv(cfg.train_csv)
        filepaths = [{"im": fname} for fname in df['Path'][:cfg.n_images]]
        filepaths = [{"im": p['im'].replace('D:/Backup/Mestrado/dados/', '/home/jovyan/imgs/').replace('\\', '/')} for p in filepaths]
        #filepaths = [{"im": p['im'].replace('D:/Backup/Mestrado/dados/', '/home/jovyan/imgs/')} for p in filepaths]
        #filepaths = [{"im": p['im'].replace('D:/Mestrado/dados/', '/home/jovyan/imgs/').replace('\\', '/')} for p in filepaths]
        #filepaths = [{"im": p['im'].replace('D:/Mestrado/dados/', '/home/jovyan/imgs/')} for p in filepaths]
        random.shuffle(filepaths)
        
        # Split into training and testing
        #test_frac = 0.1
        test_frac = 0.2
        num_ims = len(filepaths)
        num_test = int(num_ims * test_frac)
        num_train = num_ims - num_test
        train_datadict = filepaths[:num_train]
        val_datadict =  filepaths[-num_test:]
        print(f"total number of images: {num_ims}")
        print(f"number of images for training: {len(train_datadict)}")
        print(f"number of images for validation: {len(val_datadict)}")


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
                Resized(keys = ['im'], spatial_size = (128,128,128)),
                DeleteItemsd(keys=["image_meta_dict"]),  # drop all metadata entirely
                
                #SpatialPadd(
                #    keys=["im"],
                #    spatial_size=cfg.roi_size,
                #    value=0),
                
            ])


        rand_transforms = Compose([
                RandSpatialCropd(
                            keys=["im"],
                            roi_size=cfg.roi_size,
                            random_center=True,
                            random_size=False),

        ])


        #train_transforms = Compose(deterministic_transforms.transforms + rand_transforms.transforms)
        train_transforms = Compose(deterministic_transforms.transforms)
        
        train_ds = PersistentDataset(train_datadict, train_transforms, cache_dir = cfg.cache_dir)
        train_loader = DataLoader(train_ds, batch_size=cfg.train_batch_size, num_workers=cfg.num_workers, shuffle=True, prefetch_factor = cfg.prefetch_factor)
        val_ds = PersistentDataset(val_datadict, deterministic_transforms, cache_dir = cfg.cache_dir)
        val_loader = DataLoader(val_ds, batch_size=cfg.val_batch_size, num_workers=cfg.num_workers, shuffle=False,prefetch_factor = cfg.prefetch_factor)
        print('Running test')

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
