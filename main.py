import numpy as np
from numpy.fft import rfft2, fftshift
import os
from skimage import restoration
from pickle import load
from skimage.io import imread
import cv2
from createnoisyimage.createnoise import createkernelformotion as ckf
from skimage import img_as_ubyte
file = input("Please give image location: ")
img = imread(file)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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


def resmotionblur(img, angle, pixel):
    psf = ckf(pixel, angle)
    img_res0 = restoration.deconvolution.wiener(
        img[:, :, 0]/img.max(), psf, 0.001)
    img_res1 = restoration.deconvolution.wiener(
        img[:, :, 1]/img.max(), psf, 0.001)
    img_res2 = restoration.deconvolution.wiener(
        img[:, :, 2]/img.max(), psf, 0.001)
    img_res = np.dstack((img_res0, img_res1, img_res2))
    return img_as_ubyte(img_res)


model = load(open('models/modelimagetypeest.pkl', 'rb'))
scaler = load(open('models/scalerimagetypeest.pkl', 'rb'))
pixelmodel = load(open('models/modelpixelest.pkl', 'rb'))
anglemodel = load(open('models/modelangleest.pkl', 'rb'))
Ascaler = load(open('models/scalerangleest.pkl', 'rb'))
flag = True
while flag:
    feature = featureextract(img)
    X = scaler.transform([feature])
    prediction = model.predict(X)
    if prediction[0] == 0:
        img_res = cv2.bilateralFilter(img, 9, 85, 85)
        # side by side comparison
        compare = np.concatenate((img, img_res), axis=1)
        cv2.imshow('img', img_res)
        cv2.waitKey(0)
        pass
    elif prediction[0] == 1:
        img_res = cv2.medianBlur(img, 3)
        # side by side comparison
        compare = np.concatenate((img, img_res), axis=1)
        cv2.imshow('img', img_res)
        cv2.waitKey(0)
        pass
    elif prediction[0] == 2:
        feature.pop()
        feature.pop()
        feature.pop()
        X = Ascaler.transform([feature])
        angle = anglemodel.predict(X)
        pixel = pixelmodel.predict(X)-4.9
        print(pixel, angle)
        img_res = resmotionblur(img, angle, pixel)
        cv2.imshow("res", img_res)
        cv2.imshow("orginal", img)
        cv2.waitKey(0)

    else:
        img_res = img
    flag = False
    img = img_res
cv2.imwrite("out.png", img_res)
