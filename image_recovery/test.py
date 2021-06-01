from __future__ import print_function
import cv2
import numpy as np

img = cv2.imread('image_recovery/g2.jpeg')

# Salt&Pepper Recovery
# median = cv2.medianBlur(img, 3)
# compare = np.concatenate((img, median), axis=1) #side by side comparison
# cv2.imshow('img', compare)
# cv2.waitKey(0)

# # Gaussian Recovery
median = cv2.bilateralFilter(img,9,85,85)
compare = np.concatenate((img, median), axis=1) #side by side comparison
cv2.imshow('img', compare)
cv2.waitKey(0)

# Gaussian Recovery
# median = cv2.GaussianBlur(img,(7,7),0)
# compare = np.concatenate((img, median), axis=1) #side by side comparison
# cv2.imshow('img', compare)
# cv2.waitKey(0)

# Motion Blur Recovery Wiener Filter


