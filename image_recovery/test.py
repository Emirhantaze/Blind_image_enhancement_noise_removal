import cv2
import numpy as np

img = cv2.imread('g3.jpeg',cv2.IMREAD_GRAYSCALE)
median = cv2.medianBlur(img, 3)
compare = np.concatenate((img, median), axis=1) #side by side comparison

cv2.imshow('img', compare)
cv2.waitKey(0)
