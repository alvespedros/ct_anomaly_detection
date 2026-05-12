import torch
from monai.utils import set_determinism

from model import Encoder, Decoder

from dataset import get_dataloader
import os
import nibabel as nib
from collections import OrderedDict


def test_model(cfg):


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    zdim = 512
    encoder = Encoder(160, 192, 96, z_dim=zdim).to(device)
    decoder = Decoder(160, 192, 96, z_dim=zdim).to(device)


    root = 'results/z512_denseAE/'

    cpe = torch.load(os.path.join(root, 'Saved_models/encoder_best.pth'), weights_only=True)
    cpd = torch.load(os.path.join(root, 'Saved_models/decoder_best.pth'), weights_only=True)

    cpe_new = OrderedDict()
    cpd_new = OrderedDict()

    for k, v in cpe['encoder'].items():
        name = k[7:]
        cpe_new[name] = v

    for k, v in cpd['decoder'].items():
        name = k[7:]
        cpd_new[name] = v

    encoder.load_state_dict(cpe_new)
    decoder.load_state_dict(cpd_new)

    # load dataset
    save_recon_imgdir = os.path.join(root, 'recon_image')
    save_error_imgdir = os.path.join(root, 'error_map')

    if not os.path.exists(save_recon_imgdir):
        os.mkdir(save_recon_imgdir)
    if not os.path.exists(save_error_imgdir):
        os.mkdir(save_error_imgdir)

    loader = get_dataloader(cfg, mode = 'test')


    for batch_idx, data in enumerate(loader):
        print(data.keys())
        print(data['im'].shape)
        image = data['im'].to(device)
        imgname = os.path.basename(data['filename'][0])
        errorpath = os.path.join(save_error_imgdir, imgname)
        reconpath = os.path.join(save_recon_imgdir, imgname)
        affine = data['im_meta_dict']['affine'].squeeze()

        recon_image = decoder(encoder(image))

        input_np = image.cpu().numpy().squeeze()
        recon_np = recon_image.detach().cpu().numpy().squeeze()

        error_map = abs(recon_np - input_np)

        # save error map and recon image
        saveimg = nib.Nifti1Image(error_map, affine)
        nib.save(saveimg, errorpath)

        saveimg1 = nib.Nifti1Image(recon_np, affine)
        nib.save(saveimg1, reconpath)