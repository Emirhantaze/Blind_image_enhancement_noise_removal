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

ones = []
zeros = []
out = []
i = 0
j = 0
for image_col in image_cols:

    for image in image_col:
        ones.append(np.count_nonzero(image == 255))
        zeros.append(np.count_nonzero(image == 0))
        out.append(i)
        print(j)
        j += 1
    i += 1

data = {
    "Ones": ones,
    "Zeros": zeros,
    "Output": out
}
df = pd.DataFrame(data)
df.to_csv("One_Zero_info.csv")
