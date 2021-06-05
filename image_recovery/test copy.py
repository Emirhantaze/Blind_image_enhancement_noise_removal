import cv2
import numpy as np
import skimage as sk
from skimage.restoration import richardson_lucy
from numpy.random import uniform
from numpy import floor, ceil
from scipy.signal import deconvolve


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
    xlast = 0
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


img = cv2.imread('image_recovery/o2.jpeg', cv2.IMREAD_GRAYSCALE)
f = createkernelformotion(30, uniform(0, 180))
np.savetxt("a.txt", f)
img = cv2.filter2D(img, -1, f)

cv2.imshow("a", img)
# img = richardson_lucy(img, f, 20)
print(img.shape)
img, _ = deconvolve(img, f)

cv2.imshow("b", img)

cv2.waitKey(0)
