from numpy.fft import rfft2, fftshift
from cv2 import imread
import cv2
import numpy as np
import pandas as pd
from skimage.io import imread_collection
import os

start_folder = "./noise"
folders = os.listdir(start_folder)
image_cols = []
for folder in folders:
    folder = os.path.join(start_folder, folder, "*.jpg")
    image_cols.append(imread_collection(folder))


ffts0 = []
ffts1 = []
ffts2 = []

var0 = []
var1 = []
var2 = []

out = []
i = 0
j = 0
for image_col in image_cols:

    for image in image_col:
        ffimage = fftshift(np.abs(rfft2(image[:, :, 0])))
        ffimage = ffimage*255/ffimage.max()
        count = int(np.ceil(ffimage.shape[0]/9))
        elements = []
        for i in range(9):
            if i != 8:
                elements.append(np.sum(ffimage[i*count:(i+1)*count]))
            else:
                elements.append(np.sum(ffimage[i*count::]))
        elements = np.array(elements)
        elements = elements/elements.max()
        ffts0.append(elements)
        var0.append(ffimage.var())

        ffimage = fftshift(np.abs(rfft2(image[:, :, 1])))
        ffimage = ffimage*255/ffimage.max()
        count = int(np.ceil(ffimage.shape[0]/9))
        elements = []
        for i in range(9):
            if i != 8:
                elements.append(np.sum(ffimage[i*count:(i+1)*count]))
            else:
                elements.append(np.sum(ffimage[i*count::]))
        elements = np.array(elements)
        elements = elements/elements.max()
        ffts1.append(elements)
        var1.append(ffimage.var())

        ffimage = fftshift(np.abs(rfft2(image[:, :, 2])))
        ffimage = ffimage*255/ffimage.max()
        count = int(np.ceil(ffimage.shape[0]/9))
        elements = []
        for i in range(9):
            if i != 8:
                elements.append(np.sum(ffimage[i*count:(i+1)*count]))
            else:
                elements.append(np.sum(ffimage[i*count::]))
        elements = np.array(elements)
        elements = elements/elements.max()
        ffts2.append(elements)
        var2.append(ffimage.var())

        out.append(i)
        print(j)
        j += 1
    i += 1
var0 = np.array(var0)
var2 = np.array(var2)
var1 = np.array(var1)

var0 = var0/var0.max()
var1 = var1/var1.max()
var2 = var2/var2.max()


ffts1 = np.array(ffts1)
ffts2 = np.array(ffts2)
ffts0 = np.array(ffts0)

data = {
    0: ffts0[:, 0],
    1: ffts0[:, 1],
    2: ffts0[:, 2],
    3: ffts0[:, 3],
    4: ffts0[:, 4],
    5: ffts0[:, 5],
    6: ffts0[:, 6],
    7: ffts0[:, 7],
    8: ffts0[:, 8],
    9: ffts1[:, 0],
    10: ffts1[:, 1],
    11: ffts1[:, 2],
    12: ffts1[:, 3],
    13: ffts1[:, 4],
    14: ffts1[:, 5],
    15: ffts1[:, 6],
    16: ffts1[:, 7],
    17: ffts1[:, 8],
    18: ffts2[:, 0],
    19: ffts2[:, 1],
    20: ffts2[:, 2],
    21: ffts2[:, 3],
    22: ffts2[:, 4],
    23: ffts2[:, 5],
    24: ffts2[:, 6],
    25: ffts2[:, 7],
    26: ffts2[:, 8],
    27: var0,
    28: var1,
    29: var2,
    "Output": out
}
df = pd.DataFrame(data)
df.to_csv("FFT_info.csv")
