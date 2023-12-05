import numpy as np
import cv2
from scipy import signal
from skimage.metrics import structural_similarity as ssim

def dark_channel(I,r):
    """
    计算暗通道：
    参数I：输入图像
    参数r：半径
    返回值I_dark：暗通道图像
    calculate dark channel
    :param I: input image
    :param r: radius
    :return: dark channel
    """
    I=np.array(I)
    h,w,c=I.shape
    I=I.min(axis=2)
    strel=cv2.getStructuringElement(cv2.MORPH_RECT,(2*r+1,2*r+1))

    I_padding=np.ones(shape=(h+2*r, w+2*r))*255
    I_padding[r:h+r,r:w+r]=I
    I_padding[r:(r+h),r:(r+w)]=I
    I_dark_padding=np.zeros_like(I_padding)

    for i in range (r,r+h):
        for j in range(r,r+w):
            local_field=I_padding[(i-r):(i+r+1),(j-r):(j+r+1)]
            local_min=np.min(local_field[strel==1])
            I_dark_padding[i,j]=local_min

    I_dark=I_dark_padding[r:(r+h),r:(r+w)]
    return I_dark

def get_atomspheric(I):
    """
    获取大气光强：
    参数I：输入图像
    返回值A：大气光强A
    get atmospheric light
    :param I: input image
    :return: atmospheric light A
    """
    I_dark=dark_channel(I,5)
    h,w=I_dark.shape
    top_pixels=h*w//1000

    def largeset_indices(ary:np.ndarray,n):
        """
        get the n largest elements' indices
        :param ary: input numpy array
        :param n: n larget indices in array
        :return: indices
        """
        flat=ary.flatten()
        np.argpartition(flat, -n)[-n:]
        indices=np.argpartition(flat, -n)[-n:]
        indices=indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, ary.shape)

    bright_index=largeset_indices(I_dark,top_pixels)
    A=np.zeros(shape=(3,))
    A[0]=np.max(I[bright_index][0])
    A[1]=np.max(I[bright_index][1])
    A[2]=np.max(I[bright_index][2])
    return A

def get_transmission(I,A,r=5,w=0.95):
    """
    获取折射率
    参数I：输入图像
    参数A：大气光强
    参数r：半径
    参数w：omega
    返回值t：折射率
    get transmission
    :param I: input image
    :param A: atmospheric light
    :param r: radius
    :param w: omega
    :return: transmission
    """
    I=np.array(I)
    A=np.array(A)
    I_norm=I/A
    t=1-w*dark_channel(I_norm,r)
    return t

def guide_filter(I,p,r=5,eps=0.0001):
    """
    导向滤波器：
    参数I：输入图像
    参数p：折射率引导图像
    参数r：滤波器半径
    参数eps：eps正则化项
    返回值q：输出图像

    guide filter
    :param I: input image
    :param p: transmission
    :param r: radius
    :param eps: eps
    :return: filter image
    """
    mean_I=cv2.boxFilter(I,-1,(r,r))
    mean_p=cv2.boxFilter(p,-1,(r,r))
    mean_Ip=cv2.boxFilter(I*p,-1,(r,r))

    cov_Ip=mean_Ip-mean_I*mean_p

    mean_II=cv2.boxFilter(I*I,-1,(r,r))
    var_I=mean_II-mean_I*mean_I

    a=cov_Ip/(var_I+eps)
    b=mean_p-a*mean_I

    mean_a=cv2.boxFilter(a,-1,(r,r))
    mean_b=cv2.boxFilter(b,-1,(r,r))

    q=mean_a*I+mean_b
    return q

def transmission_redifine(I,t):
    """
    transmission redifine
    :param I: input image
    :param t: transmission
    :return: transmission
    """
    gray=cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
    gray=np.float64(gray)/255
    t=guide_filter(gray,t,5,0.0001)
    return t
    
def recover(I,t,A,tx=0.1):
    """
    J=(I-A)/t+A
    参数I：输入图像
    参数t：折射率
    参数A：大气光强
    参数tx：折射率阈值
    返回值res：恢复图像
    recover
    :param I: input image
    :param t: transmission
    :param A: atmospheric light
    :param tx: transmission treshold
    :return: recover image
    """
    I=np.array(I)
    res=np.empty(I.shape,I.dtype)
    t=cv2.max(t,tx)
    for index in range(0,3):
        res[:,:,index]=(I[:,:,index]-A[index])/t+A[index]
    return res


def calculate_psnr(original_image, reconstructed_image):
    # type of original_image and reconstructed_image should be float32
    original_image = original_image.astype(np.float32)
    reconstructed_image = reconstructed_image.astype(np.float32)

    # MSE
    mse = np.mean((original_image - reconstructed_image) ** 2)

    # PSNR
    max_pixel_value = 255.0
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)

    return psnr

def calculate_ssim(original_image, reconstructed_image):
    """
    calculate ssim in grayscale
    :param original_image: original image
    :param reconstructed_image: reconstructed image
    :return: ssim
    """
    # 原始图像和重建图像的数据类型应为 float32
    original_image = original_image.astype(np.float32)
    reconstructed_image = reconstructed_image.astype(np.float32)

    print(original_image.shape, reconstructed_image.shape)
    # 常量
    K1 = 0.01
    K2 = 0.03

    # 均值、方差、协方差
    mu_original = signal.convolve2d(original_image, np.ones((8, 8))/64, mode='valid')
    mu_reconstructed = signal.convolve2d(reconstructed_image, np.ones((8, 8))/64, mode='valid')

    sigma_original_sq = signal.convolve2d((original_image - mu_original)**2, np.ones((8, 8))/64, mode='valid')
    sigma_reconstructed_sq = signal.convolve2d((reconstructed_image - mu_reconstructed)**2, np.ones((8, 8))/64, mode='valid')
    sigma_original_reconstructed = signal.convolve2d((original_image - mu_original) * (reconstructed_image - mu_reconstructed), np.ones((8, 8))/64, mode='valid')

    # SSIM 公式
    numerator = (2 * mu_original * mu_reconstructed + K1) * (2 * sigma_original_reconstructed + K2)
    denominator = (mu_original**2 + mu_reconstructed**2 + K1) * (sigma_original_sq + sigma_reconstructed_sq + K2)

    ssim_map = numerator / denominator

    # 整体 SSIM
    ssim_value = np.mean(ssim_map)

    return ssim_value

def calculate_ssim_color(original_image, reconstructed_image):
    # 转换图像数据类型为浮点型
    original_image = original_image.astype(np.float64)
    reconstructed_image = reconstructed_image.astype(np.float64)

    # 计算 SSIM
    ssim_index, _ = ssim(original_image, reconstructed_image,winsize=1, full=True)

    return ssim_index


if __name__ == '__main__':
    I=cv2.imread('./image/a.jpg')
    cv2.imshow('I',I)

    I_dark=dark_channel(I,5)
    cv2.imshow('I_dark',I_dark)
    A=get_atomspheric(I)
    t=get_transmission(I,A)
    res=recover(I,t,A)
    cv2.imwrite('./image/recoverWithoutGuide.jpg',res)
    cv2.imshow('res',res)

    t=transmission_redifine(I,t)
    res=recover(I,t,A)
    cv2.imwrite('./image/recoverWithGuide.jpg',res) 
    out=cv2.imread('./image/recoverWithGuide.jpg')
    psnr = calculate_psnr(I, out)
    print("PSNR: ", psnr)
    I_gray=cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
    out_gray=cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)
    ssim_index = calculate_ssim(I_gray, out_gray)
    print("SSIM: ", ssim_index)

    cv2.imshow('res',res)

    cv2.waitKey()
