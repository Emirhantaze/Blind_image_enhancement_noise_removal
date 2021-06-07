"""
the provided links are used 
https://scikit-image.org/docs/dev/api/skimage.util.html#skimage.util.random_noise
https://stackoverflow.com/questions/34966541/how-can-one-display-an-image-using-cv2-in-python
https://stackoverflow.com/questions/14435632/impulse-gaussian-and-salt-and-pepper-noise-with-opencv

"""

from createnoise import createnoise
import cv2
from skimage.util import random_noise

img = cv2.imread("a.jpg", cv2.IMREAD_COLOR)
# cv2.imshow("aa", img)
print(img.shape)
# a = random_noise(img, mode="s&p", amount=0.05)


a = createnoise(img, type=3)

cv2.imshow(f"{a[1]}", a[0])
cv2.waitKey(0)
