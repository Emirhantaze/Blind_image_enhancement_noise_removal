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


blue_var = []
green_var = []
red_var = []
output = []
i = 0
for image_col in image_cols:

    for image in image_col:
        blue_var.append(np.var(image[:, :, 0]))
        green_var.append(np.var(image[:, :, 1]))
        red_var.append(np.var(image[:, :, 2]))
        output.append(i)

    i += 1

data = {
    "Blue_Variance": blue_var,
    "Green_Variance": green_var,
    "Red_Variance": red_var,
    "Output": output
}
df = pd.DataFrame(data)
df.to_csv("Variance_info.csv")
