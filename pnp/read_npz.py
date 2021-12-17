# ADMMを用いてdeblurringを行う
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import random
from math import sqrt, exp
from scipy import linalg
from skimage import io

from mylib.utilities import myplot
from mylib.utilities import PlotData, FigData


random.seed(0)

def main():

    start = time.time()
    
    array = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]) # sigmaに使用
    
    imgfile = ['Aerial', 'Airplane', 'Balloon', 'couple', 'Earth', 'Girl', 'Mandrill', 'Milkdrop', 'Parrots', 'Pepper']
    psnr_last = np.zeros([4, len(array)])
    # print(npz.files)

    for imgname in imgfile:
        data_psnr = np.zeros(len(array))
        npz  = np.load(os.getcwd() + '/pnp_result/np_savez_psnr_' + imgname + '.npz')
        psnr_last += npz['arr_0']/len(imgfile)
    
    # 縦:PSNR, 横:sigma^2
    fig = FigData(x_label = "$\sigma_e^2$", x_scale = "log", y_label = "PSNR"
                , savepath = "pnp_result/fig_psnr.pdf")
    plot = [
        PlotData(array, psnr_last[0], label='$\\beta = 0.1$', marker='x', linestyle='--'),
        PlotData(array, psnr_last[1], label='$\\beta = 1$', marker='*', linestyle='--'),
        PlotData(array, psnr_last[2], label='$\\beta = 10$', marker='o', linestyle='--'),
        PlotData(array, psnr_last[3], label='提案手法', marker='^', linestyle='-')
    ]
    myplot(fig, plot)

    # plt.show()

if __name__ == '__main__':
    main()