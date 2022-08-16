import cv2 as cv
import numpy as np
from skimage.util import random_noise

# Load the image
img = cv.imread("monas.jpg")

# Add salt-and-pepper noise to the image.
noise_img_snp = random_noise(img, mode='s&p',amount=0.05)

# Add gaussian noise to the image.
noise_img_gaussian = random_noise(img, mode='gaussian', mean=0, var=0.01)

# Add speckle noise to the image.
noise_img_speckle = random_noise(img, mode='speckle')

# The above function returns a floating-point image on the range [0, 1], thus we changed it to 'uint8' and from [0,255]
noise_img_snp = np.array(255*noise_img_snp, dtype = 'uint8')
noise_img_gaussian = np.array(255*noise_img_gaussian, dtype = 'uint8')
noise_img_speckle = np.array(255*noise_img_speckle, dtype = 'uint8')

# image reduction
kernel_3_3 = np.ones((3,3),np.float32)/9
img_snp_average_filter = cv.filter2D(noise_img_snp,cv.CV_8U,kernel_3_3,(-1,-1), delta = 0, borderType = cv.BORDER_DEFAULT)
img_snp_median_median = cv.medianBlur(noise_img_snp,3)
img_gaussian_average_filter = cv.filter2D(noise_img_gaussian,cv.CV_8U,kernel_3_3,(-1,-1), delta = 0, borderType = cv.BORDER_DEFAULT)
img_gaussian_median_median = cv.medianBlur(noise_img_gaussian,3)
img_speckle_average_filter = cv.filter2D(noise_img_speckle,cv.CV_8U,kernel_3_3,(-1,-1), delta = 0, borderType = cv.BORDER_DEFAULT)
img_speckle_median_median = cv.medianBlur(noise_img_speckle,3)

# Display the noise image
cv.imshow('img snp',noise_img_snp)
cv.imshow('img gaussian',noise_img_gaussian)
cv.imshow('img speckle',noise_img_speckle)

# Display the image reduction
cv.imshow('img s&p reduction with average filter', img_snp_average_filter)
cv.imshow('img s&p reduction with median filter', img_snp_median_median)
cv.imshow('img gaussian reduction with average filter', img_gaussian_average_filter)
cv.imshow('img gaussian reduction with median filter', img_gaussian_median_median)
cv.imshow('img speckle reduction with average filter', img_speckle_average_filter)
cv.imshow('img speckle reduction with median filter', img_speckle_median_median)
cv.waitKey(0)
cv.destroyAllWindows()