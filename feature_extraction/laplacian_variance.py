import cv2
import numpy as np
import pandas as pd
from skimage.io import imread_collection
import os


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


start_folder = "./noise"
folders = os.listdir(start_folder)
image_cols = []
for folder in folders:
    folder = os.path.join(start_folder, folder, "*.jpg")
    image_cols.append(imread_collection(folder))

lap_var = []
out = []
i = 0
for image_col in image_cols:

    for image in image_col:
        lap_var.append(variance_of_laplacian(image))
        out.append(i)
    i += 1

data = {
    "Laplacian_Variance": lap_var,
    "Output": out
}
df = pd.DataFrame(data)
df.to_csv("Laplacian_Variance_info.csv")
