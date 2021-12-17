import numpy as np
import os.path
import torch
from math import sqrt, exp
from scipy import linalg
from skimage.restoration import denoise_nl_means

from utils import utils_image as util
from mylib.utilities import myclip, mypsnr, conv_by_fft2

# ----------------------------------------
# ep_flag : 双対定数を徐々に増やしていく　収束が補償されるようになる
# epsig_flag : 観測誤差に対応する目的関数を 1/2||y-Hx||^2_2 から 1/2σ^2||y-Hx||^2_2 にする(基本無視でいいはず)
# resi_flag : アルゴリズムの停止条件のオンオフ 前回の推定値との残差が変動しなくなると停止　デフォルトはオン
# ----------------------------------------

def pnp_admm_deblurring_ffdnet(img, imgname, Y, H, epsilon=.01, beta=1, sigma2=0, maxit=50
                            , ep_flag = 0, epsig_flag = 0, resi_flag = 1):
    
    # ----------------------------------------
    # 準備
    # ----------------------------------------

    util.mkdir('pnp_result/ffd')
    
    residual = np.zeros(maxit) # 残差の大きさ
    
    if epsig_flag == 1: sigma = epsilon * sigma2
    else: sigma = epsilon
    
    X_tilde_old = Y.copy()
    V_tilde = Y.copy()
    U_tilde = np.zeros(Y.shape)

    # ----------------------------------------
    # FFDNetの準備
    # ----------------------------------------

    model_name = 'ffdnet_color'           # 'ffdnet_gray' | 'ffdnet_color' | 'ffdnet_color_clip' | 'ffdnet_gray_clip'
    show_img = True                     # default: False
    # show_img = False                     # default: False

    if 'color' in model_name:
        n_channels = 3        # setting for color image
        nc = 96               # setting for color image
        nb = 12               # setting for color image
    else:
        n_channels = 1        # setting for grayscale image
        nc = 64               # setting for grayscale image
        nb = 15               # setting for grayscale image
        
    model_pool = 'model_zoo'  # fixed
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

    for k in range(maxit):
        sigma_alg = sqrt(beta/epsilon)
        if ep_flag == 1: epsilon *= 1.2 # epsilon増加

        # ----------------------------------------
        # PnP
        # ----------------------------------------

        # Xの更新　変えるのはここ
        X_tilde = V_tilde - U_tilde
        X_tilde += conv_by_fft2(Y - conv_by_fft2(X_tilde, H), H, 1, sigma)
        
        # Z(ここではV)の更新
        img_L = util.uint2tensor4(X_tilde + U_tilde)
        img_L = img_L.to(device)

        sigma_map = torch.full((1,1,1,1), sigma_alg/255.).type_as(img_L)    
        img_E = model(img_L, sigma_map)
        V_tilde = util.tensor2uint(img_E)
        
        # 双対定数の更新
        U_tilde += X_tilde - V_tilde # ADMMになる
        # U_tilde = np.zeros(Y.shape) # HQSになる
        
            
        # # アルゴリズムの停止条件 ついでにpsnrの表示 いらないなら消して
        # if abs(residual[k-1] - residual[k-2]) <= .01 and abs(residual[k] - residual[k-1]) <= .01 and resi_flag == 1:
        #     print('Delta:', f'{k:02}', f'{mypsnr(img, X_tilde):.3f}', f'{residual[k]:.3f}')
        #     resi_flag = -1
        # else:
        #     print(imgname, ':', f'{k:02}', 'ffd', f'{mypsnr(img, X_tilde):.3f}')
            
            
        # if k % 10 == 0:
        #     img_name = 'ffd_' + 's' + f'{sigma2:03}' + '_' + f'{k:02}'
        #     ext = '.png'
        #     util.imsave(np.concatenate([myclip(V_tilde), myclip(X_tilde + U_tilde)], axis=1), 'pnp_result/ffd/' + img_name+ext) 

        X_tilde_old = X_tilde.copy()
        
        # ----------------------------------------
        # return
        # ----------------------------------------

        if resi_flag == -1 or k == maxit -1:
            return X_tilde, k

def denoise_nlm(Y, sigma):
    # NLMのパッチサイズ
    patch_kw = dict(patch_size=5,      # 5x5 patches
                    patch_distance=6,  # 13x13 search area
                    multichannel=True)

    X = denoise_nl_means(Y, h=sigma, sigma=sigma, **patch_kw)

    return X

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
    model_pool = 'model_zoo'  # fixed
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
