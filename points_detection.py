#!/usr/bin/env python

import cv2
import numpy as np
import glob
import argparse
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

# Initialize parser
parser = argparse.ArgumentParser(description='Detects calibration target points from a set of images.')
parser.add_argument('folder_location', type=str, help='Name of folder containing the frames.')
parser.add_argument('-th', '--threshold', type=int, metavar='N', default=128, choices=range(256), help='Value of threshold to generate binary image with all but target points filtered.')

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Initializing parameter setting using cv2.SimpleBlobDetector function
blobParams = cv2.SimpleBlobDetector_Params()

# Filter by Area
blobParams.filterByArea = True
blobParams.maxArea = 500

# Filter by Circularity
blobParams.filterByCircularity = True
blobParams.minCircularity = 0.9

# Filter by Convexity
blobParams.filterByConvexity = False
blobParams.minConvexity = 1

# Filter by Inertia
blobParams.filterByInertia = False
blobParams.minInertiaRatio = 0.01
 
# Creating a blob detector using the defined parameters
blobDetector = cv2.SimpleBlobDetector_create(blobParams)

# Import the 3D points from the csv file
obj_3D = pd.read_csv('./videos/Coords/Bundle.csv')[['X','Y','Z']]

# List of points
obj_points = []
img_points = []

# Main
args = parser.parse_args()

images = glob.glob(f'./sets/{args.folder_location}/*.jpg')
print(f"Searching images in ./sets/{args.folder_location}/")

pbar = tqdm(desc='READING FRAMES', total=len(images), unit=' frames')
total_corners = np.zeros(2)
max_corners_found = 0
max_corners_frame = ''

# Loading image
for fname in images:
    img = cv2.imread(fname, cv2.IMREAD_COLOR)
    # define the alpha and beta
    alpha = 2.0 # Contrast control
    beta = -50.0 # Brightness control

    # call convertScaleAbs function
    img_adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Applying threshold to find points
    _, thr = cv2.threshold(img_adjusted, args.threshold, 255, cv2.THRESH_BINARY_INV)

    keypoints = blobDetector.detect(thr)

    # Creating a list of corners (equivalent of findCirclesGrid)
    corners = [[[key.pt[0], key.pt[1]]] for key in keypoints]
    corners = np.array(corners, dtype=np.float32)
    # print(f"(x,y): {key.pt}, size: {key.size}, angle: {key.angle}, response: {key.response}, octave: {key.octave}, class: {key.class_id}" for key in keypoints)
    
    # print(corners.shape)
    
    if len(corners) > max_corners_found:
        max_corners_found = len(corners)
        max_corners_frame = fname

    # Drawing keypoints in the original image
    img_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_keypoints_gray = cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2GRAY)
    
    # cv2.imwrite(f'.\test{fname[7+len(folder_location):]}', img_adjusted)
    # cv2.imwrite(f'.\test{fname[7+len(folder_location):-4]}_b.jpg', thr)
    cv2.imwrite(f'./test{fname[7+len(args.folder_location):-4]}_c.jpg', img_keypoints)
    
    # cv2.imshow("First frame", img_keypoints)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    total_corners = np.append(total_corners, corners)
    
    # Recalculating corners using a better recognition algorithm
    # corners2 = cv2.cornerSubPix(img_keypoints_gray, corners, (11,11), (-1,-1), criteria)
    
    pbar.update(1)
pbar.close()

total_corners = total_corners[2:].astype(int)
total_corners = total_corners.reshape((int(len(total_corners)/2), 2))
total_corners = np.unique(total_corners, axis=0)

fs = cv2.FileStorage('./results/pts_detection/points'+args.folder_location+'.yml', cv2.FILE_STORAGE_WRITE)
fs.write('corners', total_corners)
fs.release()

img = cv2.imread(max_corners_frame, cv2.IMREAD_COLOR)
for corner in total_corners:
    cv2.circle(img, (corner[0], corner[1]), 5, (0, 0, 255), -1)
cv2.imwrite(f'./results/pts_detection/frame{args.folder_location}.jpg', img)
print(f'Frame with most points is {max_corners_frame} with {max_corners_found} found.')