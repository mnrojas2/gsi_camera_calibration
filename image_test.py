import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv.LUT(image, table)

def displayImage(img, width=1280, height=720, name='Picture'):
    # Small simple function to display images without needing to add the auxiliar functions. By default it reduces the size of the image to 1280x720.
    cv.imshow(name, cv.resize(img, (width, height)))
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def autobalance(img):
    alpha = 255 / (np.max(img)-np.min(img))
    beta = -np.min(img)*alpha
    return alpha * img + beta



img0 = cv.imread('./sets/C0131/frame52.jpg')
# h, w, _ = img0.shape
# brightness = np.sum(img0) / (255*h*w)

# min_brightness = 0.5
# alpha = brightness / min_brightness
# img0_bright = cv.convertScaleAbs(img0, alpha=alpha, beta = 255 * (1 - alpha))

# cv.imshow('Picture', img0_bright)
# cv.waitKey(0)
# cv.destroyAllWindows()

############################################
plt.figure()
plt.imshow(img0[:,:,::-1])

# img0_gamma = adjust_gamma(img0, gamma=3.0)

# plt.figure()
# plt.imshow(img0_gamma[:,:,::-1])

# img0_gray = cv.cvtColor(img0_gamma, cv.COLOR_BGR2GRAY)
# img0_balance = autobalance(img0_gray)

# plt.figure()
# plt.imshow(img0_balance)

# # Applying threshold to find points
# thr = cv.adaptiveThreshold(img0_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 51, -64)

# plt.figure()
# plt.imshow(thr)
# plt.show()
############################################

lab = cv.cvtColor(img0, cv.COLOR_BGR2LAB)

l_arr = lab[:,:,0]
a_arr = lab[:,:,1]
b_arr = lab[:,:,2]

l_gam = adjust_gamma(l_arr, gamma=3.0)

plt.figure()
plt.imshow(l_gam)

rgb_image = cv.cvtColor(lab, cv.COLOR_LAB2BGR)

plt.figure()
plt.imshow(rgb_image[:,:,::-1])

plt.show()

## Crossmatch
"""
# Initialize crossmatching algorithm functions
orb = cv.ORB_create(WTA_K=4, nfeatures=10000, edgeThreshold=31, patchSize=255)
bf = cv.BFMatcher.create(cv.NORM_HAMMING2, crossCheck=True)

img1 = cv.imread('./sets/C0131/frame53.jpg')
img1_gamma = adjust_gamma(img1, gamma=3.0)
img1_gray = cv.cvtColor(img1_gamma, cv.COLOR_BGR2GRAY)

# Detect new position of CODETARGETS
kp1, des1 = orb.detectAndCompute(img0_gray,None)
kp2, des2 = orb.detectAndCompute(img1_gray,None)

# Match descriptors.
matches = bf.match(des1,des2)
dmatches = sorted(matches, key=lambda x:x.distance)

src_pts = np.float32([kp1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)

img3 = cv.drawMatches(img0_gray,kp1,img1_gray,kp2,dmatches[:20],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
displayImage(img3)
# """
