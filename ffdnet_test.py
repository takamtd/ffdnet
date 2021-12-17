import numpy as np
import os.path
import torch
from math import sqrt, exp
from scipy import linalg
from skimage.restoration import denoise_nl_means

from utils import utils_image as util
from mylib.utilities import myclip, mypsnr, conv_by_fft2
import matplotlib.pyplot as plt
import glob

def denoise_ffdnet(Y, sigma, color=1):
    
    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    
    if color == 1:
        model_name = 'ffdnet_color'           # 'ffdnet_gray' | 'ffdnet_color' | 'ffdnet_color_clip' | 'ffdnet_gray_clip'
    if color == 0:
        model_name = 'ffdnet_color'           # 'ffdnet_gray' | 'ffdnet_color' | 'ffdnet_color_clip' | 'ffdnet_gray_clip'
    show_img = True                     # default: False
    # show_img = False                     # default: False


    task_current = 'dn'       # 'dn' for denoising | 'sr' for super-resolution
    sf = 1                    # unused for denoising
    if 'color' in model_name:
        n_channels = 3        # setting for color image
        nc = 96               # setting for color image
        nb = 12               # setting for color image
    else:
        n_channels = 1        # setting for grayscale image
        nc = 64               # setting for grayscale image
        nb = 15               # setting for grayscale image
    if 'clip' in model_name:
        use_clip = True       # clip the intensities into range of [0, 1]
    else:
        use_clip = False
    model_pool = 'pnp/model_zoo'  # fixed
    model_path = os.path.join(model_pool, model_name+'.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from models.network_ffdnet import FFDNet as net
    model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    
    # ----------------------------------------
    # 実行
    # ----------------------------------------

    img_L = util.uint2single(Y)
    img_L = util.single2tensor4(img_L)
    img_L = img_L.to(device)
    sigma = torch.full((1,1,1,1), sigma/255.).type_as(img_L)

    img_E = model(img_L, sigma)
    X = util.tensor2uint(img_E)
        
    return X

dir_path = './data'

file_names = glob.glob(dir_path+"/*")


sigma = 15
for file_name in  file_names:
    img = plt.imread(file_name)[:,:,0:3]
    if img.max() <= 1: img *= 255
    out = denoise_ffdnet(img, sigma)
    plt.imsave('result/{}_denoise.bmp'.format(os.path.splitext(file_name[len(dir_path)+1:])[0]), out)