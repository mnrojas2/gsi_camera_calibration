#!/usr/bin/env python

import os
import argparse
import cv2 as cv
import numpy as np
import glob
import re
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.spatial import distance
from sklearn.cluster import DBSCAN


# Initialize parser
parser = argparse.ArgumentParser(description='Camera calibration using GSI-based board images. It allows to find targets, save data and/or do the calibration process. Saved data can be used with "just_calibration.py".')
parser.add_argument('folder', type=str, help='Name of the folder containing the frames (*.jpg).')
parser.add_argument('-gm', '--gamma', type=float, default=False, help='Gamma correction')

def scan_for_cts(img):
    # Find contours
    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Extract centroids (X, Y positions)
    points = []
    for contour in contours:
        M = cv.moments(contour)
        if M['m00'] != 0:  # Avoid division by zero
            cx = int(M['m10'] / M['m00'])  # Calculate X position
            cy = int(M['m01'] / M['m00'])  # Calculate Y position
            points.append((cx, cy))
    points = np.array(points)
    
    # Get the clusters of points when they are closer than 50 pixels and are at least 5 points.
    db = DBSCAN(eps=50, min_samples=5).fit(points)
    labels = db.labels_
    
    # Check every cluster and save only the onest 
    cluster_data = []
    for cluster_id in set(labels):
        if cluster_id == -1:  # Skip noise points
            continue
        # Extract points in this cluster
        cluster_points = points[labels == cluster_id]
        
        # Filter clusters with maximum 10 points
        if len(cluster_points) <= 10:
            cluster_data.append(cluster_points)
            
    centroids = []
    for cluster in cluster_data:
        c_idx = distance.cdist(cluster, [cluster.mean(axis=0)]).argmin()
        centroids.append(cluster[c_idx])
    
    return np.array(centroids)

def adjust_gamma(image, gamma=1.0):
	# Adjust gamma value of the image 
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv.LUT(image, table)


# Get parser arguments
args = parser.parse_args()

# Get images from directory
frames_path = os.path.normpath(args.folder)
print(f"Searching images in {frames_path}")

images = sorted(glob.glob(f'{frames_path}/*.jpg'), key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

###############################################################################################################################
# Image processing

pbar = tqdm(desc='READING FRAMES', total=len(images), unit=' frames', dynamic_ncols=True, miniters=1)

for fname in images:
    # Read image
    img0 = cv.imread(fname)
    ffname = os.path.basename(fname)[:-4]
    
    # Detect points in image
    img_gray = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    
    img_lab = cv.cvtColor(img0, cv.COLOR_BGR2LAB)
    # Apply gamma correction
    if args.gamma:
        img_l = adjust_gamma(img_lab[:,:,0], gamma=args.gamma)
        img_gray = adjust_gamma(img_lab[:,:,0], gamma=args.gamma)
    else:
        img_l = img_lab[:,:,0]
    
    # Applying threshold to find points
    thr = cv.adaptiveThreshold(img_l, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 51, -96) #-64
    ct_centroids = scan_for_cts(thr)
    
    plt.figure()
    plt.imshow(thr)
    plt.plot(ct_centroids[:,0], ct_centroids[:,1], '.')
    plt.savefig(f'{frames_path.split('\\')[0]}/sets_code/{frames_path.split('\\')[1]}{ffname}.jpg')
    plt.close()
    
    pbar.update(1)
pbar.close()