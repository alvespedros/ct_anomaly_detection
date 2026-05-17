import nibabel as nib
import numpy as np
import os 





def save_nii(x_hat, save_path):
    predicted_img = x_hat.detach().cpu().squeeze().numpy()
    img = nib.Nifti1Image(predicted_img, np.eye(4))  # Save axis for data (just identity)
    img.header.get_xyzt_units()
    img.to_filename(save_path)
