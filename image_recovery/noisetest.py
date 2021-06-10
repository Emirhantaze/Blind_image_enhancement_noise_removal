import re
import os
import cv2
import numpy as np
from createnoise import createkernelformotion, createnoise
import matplotlib.pyplot as plt


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


img = cv2.imread("image_recovery\\o1.jpeg")
img1 = createnoise(img, 0)
img2 = createnoise(img, 1)
img3 = createnoise(img, 2)
img4 = createnoise(img, 3)
cv2.imshow


# arr = []
# for i in range(1, 100):
#     imgg = cv2.filter2D(img, -1, createkernelformotion(i, 45))
#     arr.append(variance_of_laplacian(imgg))
# # cv2.imshow(f"{variance_of_laplacian(img)}", img)
# # cv2.waitKey(0)

# start_folder = ".\\noise"
# folders = os.listdir(start_folder)

# pattern = "_(.+)_(.+)_(.+)_(.+)...."

# x = []
# y = []

# folder = "2"
# folder = os.path.join(start_folder, folder)
# i = 0
# for file in os.listdir(folder):
#     image = cv2.imread(os.path.join(folder, file))
#     meta_data = re.search(pattern, file)
#     y.append(variance_of_laplacian(image))
#     x.append(float(meta_data.group(3)))
#     print(i)
#     i += 1

# plt.scatter(x, y)
# plt.show()
