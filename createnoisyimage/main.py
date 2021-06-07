

import numpy as np
from createnoise import createnoise
import cv2
from skimage.io import imread_collection, imsave
import os
start_folder = "./kvasir-dataset-v2"
folders = os.listdir(start_folder)
image_cols = []
for folder in folders:
    folder = os.path.join(start_folder, folder, "*.jpg")
    image_cols.append(imread_collection(folder))

i = 0
for image_col in image_cols:
    for image in image_col:
        image, selectednoise, var_, amount_, pixel, angle = createnoise(image)
        imsave(os.path.join(
            "./noise", str(selectednoise), f"{i}_{var_}_{amount_}_{pixel}_{angle}.jpg"), image, quality=100)
        i += 1
        print(i/80)
