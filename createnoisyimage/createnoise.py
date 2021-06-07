from typing import Any, Tuple
from numpy import floor, ceil, round
from cv2 import filter2D
import numpy as np
from skimage.util import random_noise
from numpy.random import randint, uniform


def createnoise(img: (np.ndarray or Any), type: int = -1) -> Tuple[np.ndarray, int]:
    """
    type of noise:
                    "gaussian",
                    "s&p",
                    "motion",
                    "nothing"

    """
    if type == -1:
        selectednoise = randint(0, 4)
    else:
        selectednoise = type
    img = np.asarray(img)
    if selectednoise == 0:
        var_ = uniform(0.001, 0.1)
        pixel = 3
        amount_ = 0.001
        angle = uniform(0, 180)
        img = random_noise(img, mode="s&p", amount=amount_)
        img = filter2D(img, -1, createkernelformotion(pixel, angle))
        img = random_noise(img, var=var_)
    elif selectednoise == 1:
        pixel = 3
        amount_ = uniform(0.1, 0.001)
        angle = uniform(0, 180)
        var_ = 0.001
        img = random_noise(img, var=var_)
        img = filter2D(img, -1, createkernelformotion(pixel, angle))
        img = random_noise(img, mode="s&p", amount=amount_)
    elif selectednoise == 2:
        var_ = 0.001
        amount_ = 0.001
        img = random_noise(img, var=var_)
        img = random_noise(img, mode="s&p", amount=0.001)
        pixel = uniform(1, img.shape[0]//10)
        angle = uniform(0, 180)
        img = filter2D(
            img, -1, createkernelformotion(pixel, angle))
    elif selectednoise == 3:
        pixel = 3
        amount_ = 0.001
        angle = uniform(0, 180)
        var_ = 0.001
        img = random_noise(img, var=0.001)
        img = random_noise(img, mode="s&p", amount=0.001)
        img = filter2D(img, -1, createkernelformotion(pixel, angle))

    return img, selectednoise, var_, amount_, pixel, angle


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


print(createkernelformotion(8, 70))
