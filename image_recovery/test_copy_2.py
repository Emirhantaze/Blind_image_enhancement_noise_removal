from skimage.restoration import richardson_lucy
from scipy import fftpack
import matplotlib.pyplot as plt
import cv2
from numpy.random import uniform
import numpy as np
from numpy import floor, ceil


def pad_images_to_same_size(images):
    """
    :param images: sequence of images
    :return: list of images padded so that all images have same width and height (max width and height are used)
    """
    width_max = 0
    height_max = 0
    for img in images:
        h, w = img.shape[:2]
        width_max = max(width_max, w)
        height_max = max(height_max, h)

    images_padded = []
    for img in images:
        h, w = img.shape[:2]
        diff_vert = height_max - h
        pad_top = diff_vert//2
        pad_bottom = diff_vert - pad_top
        diff_hori = width_max - w
        pad_left = diff_hori//2
        pad_right = diff_hori - pad_left
        img_padded = cv2.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        assert img_padded.shape[:2] == (height_max, width_max)
        images_padded.append(img_padded)

    return images_padded


def createkernelformotion(lengthofkernel: int, angle: float) -> np.ndarray:
    if lengthofkernel % 2 == 1:
        lengthofkernel += 1
    lengthofkernel = int(lengthofkernel)
    kernel = np.zeros((lengthofkernel, lengthofkernel))
    angle = angle % 180
    if angle == 0:
        angle = 0.1
    flip = False
    if angle > 90:
        angle = 180-angle
        flip = True
    supposedyvalues = []
    supposedxvalues = []
    # önce sıfırdan büyük değerler için
    ylast = 0
    for i in range(1, int(lengthofkernel/2)+1):
        y = np.tan(angle*np.pi/180)*i
        if y > lengthofkernel/2:
            y = lengthofkernel/2
        for j in range(int(floor(ylast)+1), int(ceil(y)+1)):
            supposedyvalues.append(j-1)
            supposedxvalues.append(i-1)
        ylast = round(y, 2)

        if ylast == lengthofkernel/2:
            break
    l = len(supposedxvalues)
    for i in range(l):
        supposedxvalues.append(-supposedxvalues[i]-1)
        supposedyvalues.append(-supposedyvalues[i]-1)

    supposedxvalues = np.array(
        np.array(supposedxvalues)+lengthofkernel/2, dtype=int)
    supposedyvalues = np.array(
        np.array(supposedyvalues)+lengthofkernel/2, dtype=int)
    if l == 0:
        l = 1
    kernel[supposedyvalues, supposedxvalues] = 0.5/(l)
    if flip:
        kernel = np.flip(kernel, 0)
    return kernel


def convolve(star, psf):
    star_fft = fftpack.fftshift(fftpack.fftn(star))
    psf_fft = fftpack.fftshift(fftpack.fftn(psf))
    return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(star_fft*psf_fft)))


def deconvolve(star, psf):
    star_fft = fftpack.fftshift(fftpack.fftn(star))
    psf_fft = fftpack.fftshift(fftpack.fftn(psf))
    return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(star_fft/psf_fft)))


star = cv2.imread('image_recovery/o2.jpeg', cv2.IMREAD_GRAYSCALE)
psf = createkernelformotion(300, uniform(0, 180))

star, psf = pad_images_to_same_size([star, psf])
psf = psf+(((psf == 0))*1e-10)
star_conv = convolve(star, psf)
star_deconv = richardson_lucy(star_conv, psf, 50)
# star_deconv = deconvolve(star_conv, psf)

f, axes = plt.subplots(2, 2)
axes[0, 0].imshow(star)
axes[0, 1].imshow(psf)
axes[1, 0].imshow(np.real(star_conv))
axes[1, 1].imshow(np.real(star_deconv))
plt.show()
