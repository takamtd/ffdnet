
# ----------------------------------------
# PnPを実行するだけ
# ----------------------------------------

import time
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from math import sqrt

from mylib.pnp import pnp_admm_deblurring_ffdnet
from mylib.utilities import myclip, mypsnr, conv_by_fft2, myplot

from utils import utils_image as util

def main():

    start = time.time()
    
    # ----------------------------------------
    # 準備
    # ----------------------------------------

    iter = 1 # 複数回実行して平均取りたいなら
    sigma2 = 5 # 複数のsigma^2 (加算ノイズの分散)
    # array = np.array([2]) # 
    maxit = 50 # 最大繰り返し数

    H = plt.imread("ker/k_large_1.png")[:,:]; H = H / np.sum(H)
    # H = plt.imread("ker/ker_4.png")[:,:, 0]; H = H / np.sum(H)
    
    img = plt.imread("images/Pepper.bmp")[:,:,0:3]
    if img.max() <= 1: img *= 255

    x_rt = np.empty(img.shape)

    # ----------------------------------------
    # 画像のブラー, ノイズ付加 (劣化画像を作る)
    # ----------------------------------------

    Y_clear = conv_by_fft2(img, H) # 画像のブラー
    noise = np.random.normal(loc = 0, scale = sqrt(sigma2), size = img.shape) # sigmaでノイズ生成
    Y = Y_clear + noise # ブラー画像にノイズを不可

    # ----------------------------------------
    # 実行
    # ----------------------------------------

    beta = 1
    x_rt, _ = pnp_admm_deblurring_ffdnet(img, "imgname", Y, H, beta=beta, sigma2=sigma2
                , ep_flag = 0, epsig_flag = 0)
    
    # util.imshow(np.concatenate([myclip(img),myclip(x_rt)], axis=1))

    # ----------------------------------------
    # 経過時間
    # ----------------------------------------

    elapsed_time = time.time() - start
    print ("elapsed_time:" + f"{elapsed_time:.2f}" + "[sec]")
    
    
if __name__ == "__main__":
    main()