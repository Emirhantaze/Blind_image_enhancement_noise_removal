import cv2
import numpy as np

img = cv2.imread('image_recovery/s2.jpeg')

# # Salt&Pepper Recovery
median = cv2.medianBlur(img, 3)
compare = np.concatenate((img, median), axis=1) #side by side comparison
cv2.imshow('img', compare)
cv2.waitKey(0)

# # Gaussian Recovery
# median = cv2.bilateralFilter(img,9,85,85)
# compare = np.concatenate((img, median), axis=1) #side by side comparison
# cv2.imshow('img', compare)
# cv2.waitKey(0)

# Gaussian Recovery
# median = cv2.GaussianBlur(img,(7,7),0)
# compare = np.concatenate((img, median), axis=1) #side by side comparison
# cv2.imshow('img', compare)
# cv2.waitKey(0)

# Motion Blur Recovery Wiener Filter

# import os
# import numpy as np
# from numpy.fft import fft2, ifft2
# from scipy.signal import gaussian, convolve2d
# import matplotlib.pyplot as plt


# def wiener_filter(img, kernel, K):
# 	kernel /= np.sum(kernel)
# 	dummy = np.copy(img)
# 	dummy = fft2(dummy)
# 	kernel = fft2(kernel, s = img.shape)
# 	kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
# 	dummy = dummy * kernel
# 	dummy = np.abs(ifft2(dummy))
# 	return dummy

# def gaussian_kernel(kernel_size = 3):
# 	h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
# 	h = np.dot(h, h.transpose())
# 	h /= np.sum(h)
# 	return h
# img = cv2.imread('image_recovery/m2.jpeg',cv2.IMREAD_GRAYSCALE)
# noisy_img = cv2.imread('image_recovery/m2.jpeg',cv2.IMREAD_GRAYSCALE)
# kernel = gaussian_kernel(3)
# median = wiener_filter(noisy_img, kernel, K = 10)
# compare = np.concatenate((img, median), axis=1)
# cv2.imshow('img', compare)
# cv2.waitKey(0)


