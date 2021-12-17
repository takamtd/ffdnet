import numpy as np
from scipy import fftpack
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt



# ----------------------------------------
# plotするデータについて
# ----------------------------------------

class PlotData():
    def __init__(self, x_axis, y_axis, label = "", marker = ".", linestyle = "-"):
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.label = label
        self.marker = marker
        self.linestyle = linestyle

# ----------------------------------------
# plotで表示する図について
# ----------------------------------------

class FigData():
    def __init__(self, x_label = "", x_fontsize = 18, x_scale = "linear"
                , y_label = "", y_fontsize = 18, y_scale = "linear"
                , savepath = None):
        self.x_label  = x_label
        self.y_label = y_label
        self.x_fontsize = x_fontsize
        self.y_fontsize = y_fontsize
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.savepath = savepath

# ----------------------------------------
# plotを行う PlotDataは複数入力可
# ----------------------------------------

def myplot(FigData, PlotDatas, plt_show = False):
    plt.figure()
    for p in PlotDatas:
        plt.plot(p.x_axis, p.y_axis, label=p.label, marker=p.marker, linestyle=p.linestyle)
    plt.xlabel(FigData.x_label, fontsize=FigData.x_fontsize)
    plt.ylabel(FigData.y_label, fontsize=FigData.y_fontsize)
    plt.xscale(FigData.x_scale)
    plt.yscale(FigData.y_scale)
    plt.grid() 
    plt.tick_params(labelsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.savefig(FigData.savepath, dpi = 200)  if FigData.savepath is not None else None
    plt.show() if plt_show is True else None
    

# ----------------------------------------
# 画像を0~255のint型にクリップ
# ----------------------------------------

def myclip(x):
    return np.clip(x, 0, 255).astype(int)

# ----------------------------------------
# 原画像とのPSNRを返す
# ----------------------------------------

def mypsnr(img, x):
    return peak_signal_noise_ratio(img, myclip(x), data_range = 255)

# ----------------------------------------
# 画像とカーネルを畳み込み
# ----------------------------------------

def conv_by_fft2(A, B, flag = 0, epsilon = 0, multichannel = 1):
    # flag = 0: A*B (AとBを畳み込み)
    # flag = 1: A*B^{\dag} (AとBの逆行列を畳み込み)

    w, h = A.shape[0:2]
    wb,hb = B.shape[0:2]

    sz = (w - wb, h - hb)
    bigB = np.pad(B, (((sz[0]+1)//2, sz[0]//2), ((sz[1]+1)//2, sz[1]//2)), 'constant')
    bigB = fftpack.ifftshift(bigB)
    fft2B = fftpack.fft2(bigB)

    if multichannel == 1: # color
        if flag == 1:
            fft2B = fft2B.conjugate() / (abs(fft2B) **2 + epsilon*np.ones([w,h])) 
        C = np.empty(A.shape)
        for i in range(3):
            C[:,:,i] = np.real(fftpack.ifft2(fftpack.fft2(A[:,:,i]) * fft2B))
        return C

    else: # single channel
        if flag == 1:
            fft2B = fft2B.conjugate() / (abs(fft2B) **2 + epsilon*np.ones([w,h]))  
        return np.real(fftpack.ifft2(fftpack.fft2(A) * fft2B))

# ----------------------------------------
# ガウシアンカーネルの作成
# ----------------------------------------
        
def gaussian_kernel(ksize = 3):
    combs = [1]

    for i in range(1, ksize):
        ratio = (ksize-i)/(i)
        combs.append(combs[-1]*ratio)

    combs = np.array(combs).reshape(1,ksize)/(2**(ksize-1))
    return combs.T.dot(combs)