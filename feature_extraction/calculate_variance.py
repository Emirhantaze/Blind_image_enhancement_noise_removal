from matplotlib import pyplot
import numpy as np
import pandas as pd
from skimage.io import imread
import os
import re
from ctypes import wintypes, windll
from functools import cmp_to_key

# function added from stackoverflow https://stackoverflow.com/a/48030307


def winsort(data):
    _StrCmpLogicalW = windll.Shlwapi.StrCmpLogicalW
    _StrCmpLogicalW.argtypes = [wintypes.LPWSTR, wintypes.LPWSTR]
    _StrCmpLogicalW.restype = wintypes.INT

    def cmp_fnc(psz1, psz2): return _StrCmpLogicalW(psz1, psz2)
    return sorted(data, key=cmp_to_key(cmp_fnc))


start_folder = ".\\noise"
folders = os.listdir(start_folder)


blue_var = []
green_var = []
red_var = []
output = []
var_ = []
amount_ = []
pixel = []
angle = []
i = 0
pattern = "_(.+)_(.+)_(.+)_(.+)...."

for folder in folders:
    j = 0
    folder = os.path.join(start_folder, folder)
    for file in os.listdir(folder):
        image = imread(os.path.join(folder, file))
        blue_var.append(np.var(image[:, :, 0]))
        green_var.append(np.var(image[:, :, 1]))
        red_var.append(np.var(image[:, :, 2]))
        output.append(i)
        meta_data = re.search(pattern, file)
        var_.append(float(meta_data.group(1)))
        amount_.append(float(meta_data.group(2)))
        pixel.append(float(meta_data.group(3)))
        angle.append(float(meta_data.group(4)))
        print(i, j)
        j += 1
    i += 1


data = {
    "Blue_Variance": blue_var,
    "Green_Variance": green_var,
    "Red_Variance": red_var,
    "Output": output,
    "Var_": var_, "Amount": amount_, "Pixel": pixel, "Angle": angle}
df = pd.DataFrame(data)
df.to_csv("Variance_info.csv")
