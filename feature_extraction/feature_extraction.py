import numpy as np
import cv2
img = cv2.imread("noise/0/2.jpg", cv2.IMREAD_COLOR)

print(np.var(img[:, :, 0]))
print(np.var(img[:, :, 1]))
print(np.var(img[:, :, 2]))

img = cv2.imread("noise/1/3.jpg", cv2.IMREAD_COLOR)

print(np.var(img[:, :, 0]))
print(np.var(img[:, :, 1]))
print(np.var(img[:, :, 2]))

img = cv2.imread("noise/2/1.jpg", cv2.IMREAD_COLOR)

print(np.var(img[:, :, 0]))
print(np.var(img[:, :, 1]))
print(np.var(img[:, :, 2]))

img = cv2.imread("noise/3/6.jpg", cv2.IMREAD_COLOR)

print(np.var(img[:, :, 0]))
print(np.var(img[:, :, 1]))
print(np.var(img[:, :, 2]))
