#!/usr/bin/env python

import os
import re
import copy
import glob
import argparse
import numpy as np
import pandas as pd
import cv2 as cv
import pymap3d as pm

from matplotlib import pyplot as plt

# Initialize parser
parser = argparse.ArgumentParser(description='Camera calibration using chessboard images.')
parser.add_argument('file', type=str, help='Name of the file containing targets in 3D data (*.txt).')
parser.add_argument('-cb', '--calibfile', type=str, metavar='file', help='Name of the file containing calibration results (*.yml), for point reprojection and/or initial guess during calibration.')
parser.add_argument('-dg', '--displaygraphs', action='store_true', default=False, help='Shows graphs and images.')


# Functions
def displayImage(img, width=1280, height=720, name='Picture'):
    # Small simple function to display images without needing to add the auxiliar functions. By default it reduces the size of the image to 1280x720.
    cv.imshow(name, cv.resize(img, (width, height)))
    cv.waitKey(0)
    cv.destroyAllWindows()
        
def displayImageWPoints(img, *args, name='Image', show_names=False, save=False, fdir='new_set'):
    # Create output folder if it wasn't created yet
    if img.ndim == 2:
        img_copy = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    else:
        img_copy = copy.copy(img)
    for arg in args:
        if isinstance(arg, np.ndarray):
            values = arg.reshape(-1,2).astype(int)
        elif isinstance(arg, pd.DataFrame):
            keys = arg.index.to_list()
            values = arg.to_numpy().astype(int)
        else:
            raise TypeError('Argument format is not allowed.')
        clr = [255, 0, 0]
        if len(args) > 1:
            clr = (np.array([128, 128, 128]) + np.random.randint(-128, 128, size=3)).tolist()
        for i in range(arg.shape[0]):
            cv.circle(img_copy, values[i], 4, clr, -1)
            if show_names and isinstance(arg, pd.DataFrame):
                cv.putText(img_copy, keys[i], values[i], cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    if save:
        if not os.path.exists('./tests/tracked-sets/'+fdir):
            os.mkdir('./tests/tracked-sets/'+fdir)
        cv.imwrite(f'./tests/tracked-sets/{fdir}/{name}.jpg', img_copy)
    else:
        displayImage(img_copy, name=name)
    
def scatterPlot(*args, name='Image'):
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    for arg in args:
        ax.scatter(arg[:,0], -arg[:,1])
    plt.get_current_fig_manager().set_window_title(name)
    plt.show()

# Main

# Get parser arguments
args = parser.parse_args()

# GSI data import
# Import the 3D points from the csv file
gpsArr = pd.read_csv(f'./datasets/April22GPS/{args.file}.txt', header=None, names=['Lat','Lon','Alt', 'null'])
gpsArr.dropna(axis=1, inplace=True)
gpsArr.drop(['pt2', 'pt12'], inplace=True)

points_3D = gpsArr.to_numpy() # BEWARE: to_numpy() doesn't generate a copy but another instance to access the same data. So, if points_3D changes, obj3D will too.

# Point of interest (center)
POI = gpsArr.loc[['pt5']].to_numpy()[0]

# Load .yml file
print(f'Loading {args.calibfile}.yml')
fs = cv.FileStorage(f'./results/{args.calibfile}.yml', cv.FILE_STORAGE_READ)
print(f"File '{args.calibfile}.yml' description:",fs.getNode("summary").string())

# Camera intrinsic parameters for calibrations
mtx = fs.getNode("camera_matrix").mat()
dist_coeff = fs.getNode("dist_coeff").mat()

# Points 2D
points_2D = np.array([
    [[2171,  220]],
    [[2191,  884]],
    [[2071, 1303]],
    [[1833,  750]],
    [[1937, 1444]],
    [[1714, 1293]],
    [[1475, 1784]],
    [[1184, 1081]],
    [[1048, 1599]],
    [[1103,  315]],
], dtype=np.float64)
# Note: solvePnP will fail if values in points2D aren't in float format.

new_pnts = []
for i in range(points_3D.shape[0]):
    pnts = pm.geodetic2enu(points_3D[i,0], points_3D[i,1], points_3D[i,2], POI[0], POI[1], POI[2])
    new_pnts.append(pnts)
pts3D = np.array(new_pnts)

if args.displaygraphs:
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts3D[:,0], pts3D[:,1], pts3D[:,2], zdir='z')
    for i in range(pts3D.shape[0]):
        ax.text(pts3D[i,0],pts3D[i,1],pts3D[i,2],  '%s' % (str(i+1)), size=20, zorder=1, color='k')
    ax.set_aspect('equal')
    plt.show()

# Get images from directory
print(f"Searching images in ./sets/C0040/")
images = sorted(glob.glob(f'./sets/C0040/*.jpg'), key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

fname = images[0]

img = cv.imread(fname)
ffname = fname.split("\\")[-1][:-4]

if args.displaygraphs:
    displayImageWPoints(img, points_2D, name=ffname)
     
_, rvec, tvec = cv.solvePnP(pts3D, points_2D, cameraMatrix=mtx, distCoeffs=dist_coeff, flags=cv.SOLVEPNP_SQPNP)
repr_pts2D = cv.projectPoints(pts3D, rvec, tvec, cameraMatrix=mtx, distCoeffs=dist_coeff)[0]

if args.displaygraphs:
    displayImageWPoints(img, repr_pts2D, name=ffname)
    
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.medianBlur(gray, 5)
thr = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, -32)

new_pts2D = []
for pt in repr_pts2D:
    npt = pt.reshape(2).astype(int)
    wdw = 25
    contours, _ = cv.findContours(thr[npt[1]-wdw:npt[1]+wdw,npt[0]-wdw:npt[0]+wdw],cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    # Calculate moments
    M = cv.moments(contours[-1])
    
    # Calculate x,y
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    new_pts2D.append([[npt[0]-wdw+cX, npt[1]-wdw+cY]])

new_pts2D = np.array(new_pts2D)

displayImageWPoints(img, new_pts2D, name=ffname)