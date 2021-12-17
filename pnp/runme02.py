
# ----------------------------------------
# PnPを用いてぶれ除去　複数のシグマ　2つ比較
# ----------------------------------------

import time
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from math import sqrt

from mylib.pnp import pnp_admm_deblurring_ffdnet
from mylib.utilities import myclip, mypsnr, conv_by_fft2, myplot
from mylib.utilities import PlotData, FigData

from utils import utils_image as util

def main():

    start = time.time()
    
    # ----------------------------------------
    # 準備
    # ----------------------------------------

    iter = 1 # 複数回実行して平均取りたいなら
    array = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]) # 複数のsigma^2 (加算ノイズの分散)
    # array = np.array([2]) # 
    maxit = 50 # 最大繰り返し数

    # ----------------------------------------
    # 使用カーネル
    # ----------------------------------------

    H = plt.imread("ker/k_large_1.png")[:,:]; H = H / np.sum(H)
    # H = plt.imread("ker/ker_4.png")[:,:, 0]; H = H / np.sum(H)

    # ----------------------------------------
    # 画像の入力
    # ----------------------------------------

    # imgfile = ["Aerial", "Airplane", "Balloon", "couple", "Earth", "Girl", "Mandrill", "Milkdrop", "Parrots", "Pepper"]
    imgfile = ["Pepper"]
    
    for imgname in imgfile:
        img = plt.imread("images/" + imgname + ".bmp")[:,:,0:3]
        if img.max() <= 1: img *= 255

        x_rt = np.empty([2, img.shape[0],  img.shape[1],  img.shape[2]])
        iter_rt = np.empty(2)
        
        psnr_last = np.zeros([2, len(array)])
        iter_last = np.zeros([2, len(array)])

        # ----------------------------------------
        # 実行
        # ----------------------------------------

        for i in range(iter):

            for k in range(len(array)):

                # ----------------------------------------
                # 画像のブラー, ノイズ付加 (劣化画像を作る)
                # ----------------------------------------

                Y_clear = conv_by_fft2(img, H) # 画像のブラー
                sigma2 = array[k]
                noise = np.random.normal(loc = 0, scale = sqrt(sigma2), size = img.shape) # sigmaでノイズ生成
                Y = Y_clear + noise # ブラー画像にノイズを不可

                print("*** iter:", i+1, "sigma:", f"{sqrt(sigma2):.2f}", "***")

                # ----------------------------------------
                # 実行
                # ----------------------------------------

                # 1: 
                beta = 1
                x_rt[0], iter_rt[0] = pnp_admm_deblurring_ffdnet(img, imgname, Y, H, beta=beta, sigma2=sigma2
                            , ep_flag = 0, epsig_flag = 0)
                # 2: 
                beta = 10
                x_rt[1], iter_rt[1] = pnp_admm_deblurring_ffdnet(img, imgname, Y, H, beta=beta, sigma2=sigma2
                            , ep_flag = 0, epsig_flag = 0)
                
                img_name = "s" + f"{sigma2:03}"
                ext = ".png"
                util.imsave(np.concatenate([np.concatenate([myclip(img),myclip(Y)], axis=0), np.concatenate([myclip(x_rt[0]), myclip(x_rt[1])],axis=0)], axis=1), "pnp_result/" + img_name+ext)
                
                # ----------------------------------------
                # psnr, 停止ステップ数を保存
                # ----------------------------------------

                for j in range(0,2):
                    psnr_last[j, k] += mypsnr(img, x_rt[j])/iter
                    iter_last[j, k] += iter_rt[j]/iter
        
        # ----------------------------------------
        # グラフ
        # ----------------------------------------

        # 縦:PSNR, 横:sigma^2
        fig = FigData(x_label = "$\sigma_e^2$", x_scale = "log", y_label = "PSNR"
                    , savepath = "pnp_result/fig_psnr.png")
        plot = [
            PlotData(array, psnr_last[0], label="1", marker="x", linestyle="--"),
            PlotData(array, psnr_last[1], label="2", marker="*", linestyle="--")
        ]
        myplot(fig, plot)

        # 縦:停止ステップ数, 横:sigma^2
        fig = FigData(x_label = "$\sigma_e^2$", x_scale = "log", y_label = "k"
                    , savepath = "pnp_result/fig_iter.png")
        plot = [
            PlotData(array, iter_last[0], label="1", marker="x"),
            PlotData(array, iter_last[1], label="2", marker="*") 
        ]
        myplot(fig, plot)
        
        # ----------------------------------------
        # npzで保存したいなら
        # ----------------------------------------
        
        # np.savez("pnp_result/np_savez_psnr_" + imgname, psnr_last)
        # np.savez("pnp_result/np_savez_resi_" + imgname, iter_last)

    # ----------------------------------------
    # 経過時間
    # ----------------------------------------

    elapsed_time = time.time() - start
    print ("elapsed_time:" + f"{elapsed_time:.2f}" + "[sec]")

    plt.show()
    
    
    

if __name__ == "__main__":
    main()