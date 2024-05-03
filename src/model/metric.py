import torch
import numpy as np
import math
from skimage.measure.simple_metrics import compare_psnr
from skimage.measure  import compare_ssim
# from skimage.metrics import structural_similarity as compare_ssim
#from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    
    
    
def psnr(img, imclean):
    img = img.mul(255).clamp(0, 255).round().div(255)
    imclean = imclean.mul(255).clamp(0, 255).round().div(255)
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = []
    for i in range(Img.shape[0]):
        ps = compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=1.0)
        if np.isinf(ps):
            continue
        PSNR.append(ps)
    return sum(PSNR)/len(PSNR)


def ssim(img, imclean):
    img = img.mul(255).clamp(0, 255).round().div(255)
    imclean = imclean.mul(255).clamp(0, 255).round().div(255)
    Img = img.permute(0, 2, 3, 1).data.cpu().numpy().astype(np.float32)
    Iclean = imclean.permute(0, 2, 3, 1).data.cpu().numpy().astype(np.float32)
    SSIM = []
    for i in range(Img.shape[0]):
        ss = compare_ssim(Iclean[i,:,:,:], Img[i,:,:,:], multichannel =True ) #multichannel =True  data_range=1,channel_axis=-1
        SSIM.append(ss) 
    return sum(SSIM)/len(SSIM)

