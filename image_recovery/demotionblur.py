from skimage import restoration
import numpy as np
from cv2 import imread
import cv2
from numpy import savetxt
from createnoise import createkernelformotion as ckf
from cv2 import filter2D
from scipy.signal import convolve2d


img = imread(
    "2_0.001_0.001_15.467834093743738_24.88727279720132.jpg")
psf = ckf(15.467834093743738, 24.88727279720132)
# s = img.shape


# img_blurred = convolve2d(img, psf, "auto")

# img_blurred = filter2D(img, -1, psf)
# img_blurred = img_blurred/img_blurred.max()
# cv2.imshow("q", img_blurred)


# img_res = restoration.wiener(img_blurred, psf, 0.00001)


img_res, _ = restoration.deconvolution.unsupervised_wiener(
    img[:, :, 1]/img.max(), psf)


print(img_res[:].max())
cv2.imshow("w", img_res[:])

cv2.waitKey(0)
