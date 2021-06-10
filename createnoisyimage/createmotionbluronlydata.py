import numpy as np
from skimage.io import imread, imsave
import os
from ctypes import wintypes, windll
from functools import cmp_to_key
from createnoise import createkernelformotion as ckf
from numpy.fft import rfft2, fftshift
import cv2
from numpy.random import uniform
from numpy import save


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def featureextract(img):
    feature = [img[:, :, 0].var(), img[:, :, 1].var(
    ), img[:, :, 2].var(), variance_of_laplacian(img)]
    one_count = np.sum(img == 255)
    zero_count = np.sum(img == 0)
    feature.append(one_count)
    feature.append(zero_count)

    ffimage = fftshift(np.abs(rfft2(img[:, :, 0])))
    ffimage = ffimage*255/ffimage.max()
    count = int(np.ceil(ffimage.shape[0]/9))

    for i in range(9):
        if i != 8:
            feature.append(np.sum(ffimage[i*count:(i+1)*count]))
        else:
            feature.append(np.sum(ffimage[i*count::]))
    a = np.array(feature[len(feature)-9::])
    a = a/a.max()
    for i in range(len(feature)-9, len(feature)):
        feature[i] = a[i-(len(feature)-9)]

    ffimage = fftshift(np.abs(rfft2(img[:, :, 1])))
    ffimage = ffimage*255/ffimage.max()
    count = int(np.ceil(ffimage.shape[0]/9))

    for i in range(9):
        if i != 8:
            feature.append(np.sum(ffimage[i*count:(i+1)*count]))
        else:
            feature.append(np.sum(ffimage[i*count::]))
    a = np.array(feature[len(feature)-9::])
    a = a/a.max()
    for i in range(len(feature)-9, len(feature)):
        feature[i] = a[i-(len(feature)-9)]

    ffimage = fftshift(np.abs(rfft2(img[:, :, 2])))
    ffimage = ffimage*255/ffimage.max()
    count = int(np.ceil(ffimage.shape[0]/9))

    for i in range(9):
        if i != 8:
            feature.append(np.sum(ffimage[i*count:(i+1)*count]))
        else:
            feature.append(np.sum(ffimage[i*count::]))
    a = np.array(feature[len(feature)-9::])
    a = a/a.max()
    for i in range(len(feature)-9, len(feature)):
        feature[i] = a[i-(len(feature)-9)]
    feature.append(0.5)
    feature.append(0.5)
    feature.append(0.5)
    return feature


def winsort(data):
    _StrCmpLogicalW = windll.Shlwapi.StrCmpLogicalW
    _StrCmpLogicalW.argtypes = [wintypes.LPWSTR, wintypes.LPWSTR]
    _StrCmpLogicalW.restype = wintypes.INT

    def cmp_fnc(psz1, psz2): return _StrCmpLogicalW(psz1, psz2)
    return sorted(data, key=cmp_to_key(cmp_fnc))


start_folder = ".\\kvasir-dataset-v2"
folders = os.listdir(start_folder)
j = 0

features = []
pixels = []
angles = []
for folder in folders:
    folder = os.path.join(start_folder, folder)
    for file in winsort(os.listdir(folder)):
        image = imread(os.path.join(folder, file))
        pixel = uniform(1, image.shape[0]//10)
        angle = uniform(0, 180)
        img = cv2.filter2D(image, -1, ckf(pixel, angle))
        feature = featureextract(img)
        features.append(feature)
        j += 1
        print(j/(80))


save("features.npy", np.array(features))
save("angles.npy", np.array(angles))
save("pixels.npy", np.array(pixels))
