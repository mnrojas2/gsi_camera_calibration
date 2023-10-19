#!/usr/bin/env python

import cv2 as cv
import numpy as np
import glob
import argparse
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

# Initialize parser
parser = argparse.ArgumentParser(description='Detects calibration target points from a set of images.')
parser.add_argument('folder', type=str, help='Name of folder containing the frames.')
parser.add_argument('-th', '--threshold', type=int, metavar='N', default=128, choices=range(256), help='Value of threshold to generate binary image with all but target points filtered.')

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Initializing parameter setting using cv.SimpleBlobDetector function
blobParams = cv.SimpleBlobDetector_Params()

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
blobDetector = cv.SimpleBlobDetector_create(blobParams)

# Import the 3D points from the csv file
obj_3D = pd.read_csv('./datasets/coords/Bundle.csv')[['X','Y','Z']]

# List of points
obj_points = []
img_points = []

# Main
args = parser.parse_args()

images = glob.glob(f'./sets/{args.folder}/*.jpg')
print(f"Searching images in ./sets/{args.folder}/")

pbar = tqdm(desc='READING FRAMES', total=len(images), unit=' frames')
total_corners = np.zeros(2)
max_corners_found = 0
max_corners_frame = ''

# Loading image
for fname in images:
    img = cv.imread(fname, cv.IMREAD_COLOR)
    ffname = fname[8+len(args.folder):-4]
    # define the alpha and beta
    alpha = 2.0 # Contrast control
    beta = -50.0 # Brightness control

    # call convertScaleAbs function
    img_adjusted = cv.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Applying threshold to find points
    _, thr = cv.threshold(img_adjusted, args.threshold, 255, cv.THRESH_BINARY_INV)

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
    img_keypoints = cv.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_keypoints_gray = cv.cvtColor(img_keypoints, cv.COLOR_BGR2GRAY)
    
    # cv.imwrite(f'.\test{fname[7+len(folder_location):]}', img_adjusted)
    # cv.imwrite(f'.\test{fname[7+len(folder_location):-4]}_b.jpg', thr)
    # cv.imwrite(f'./tests/{ffname}_c.jpg', img_keypoints)
    
    # cv.imshow("First frame", img_keypoints)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    
    total_corners = np.append(total_corners, corners)
    
    # Recalculating corners using a better recognition algorithm
    # corners2 = cv.cornerSubPix(img_keypoints_gray, corners, (11,11), (-1,-1), criteria)
    
    pbar.update(1)
pbar.close()

total_corners = total_corners[2:].astype(int)
total_corners = total_corners.reshape((int(len(total_corners)/2), 2))
total_corners = np.unique(total_corners, axis=0)

fs = cv.FileStorage('./results/pts_detection/points'+args.folder+'.yml', cv.FILE_STORAGE_WRITE)
fs.write('corners', total_corners)
fs.release()

img = cv.imread(max_corners_frame, cv.IMREAD_COLOR)
for corner in total_corners:
    cv.circle(img, (corner[0], corner[1]), 5, (0, 0, 255), -1)
cv.imwrite(f'./results/pts_detection/frame{args.folder}.jpg', img)
print(f'Frame with most points is {max_corners_frame} with {max_corners_found} found.')