#!/usr/bin/env python

import os
import re
import copy
import glob
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import cv2 as cv
import pymap3d as pm
from tqdm import tqdm
from matplotlib import pyplot as plt

# Initialize parser
parser = argparse.ArgumentParser(description='Camera calibration using chessboard images.')
parser.add_argument('file', type=str, help='Name of the file containing targets in 3D data (*.txt).')
parser.add_argument('-cb', '--calibfile', type=str, metavar='file', help='Name of the file containing calibration results (*.yml), for point reprojection and/or initial guess during calibration.')
parser.add_argument('-dg', '--displaygraphs', action='store_true', default=False, help='Shows graphs and images.')
parser.add_argument('-s',  '--save', action='store_true', default=False, help='Saves data related with point location and reconstruction vectors.')
parser.add_argument('-hf', '--halfway', action='store_true', default=False, help='Name of the file containing target data to restart tracking process from any frame (*.txt).')


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

# Crossmatch
# Initialize crossmatching algorithm functions
orb = cv.ORB_create()
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

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

points_2D = np.array([
    [[2188,  281]],
    [[2211,  941]],
    [[2091, 1361]],
    [[1850,  808]],
    [[1955, 1501]],
    [[1731, 1350]],
    [[1488, 1844]],
    [[1197, 1139]],
    [[1061, 1657]],
    [[1121,  383]]
], dtype=np.float64)
# Note: solvePnP will fail if values in points2D aren't in float format.

st_frame = 0
points2D_all = []
ret_names = []
vecs = []

if args.halfway:
    # Load .txt file with some specific frame codetarget locations
    print(f'./datasets/dataTOCO_vid-1.pkl')
    pFile = pickle.load(open(f"./datasets/dataTOCO_vid-1.pkl","rb"))
    
    points2D_all = pFile['2D_points']
    ret_names = pFile['frame_name']
    vecs = pFile['rt_vectors']
    points_2D = np.array(points2D_all[-1])
    
    # Save starting point
    st_frame = 1+int(pFile['last_passed_frame'])

new_pt3D = []
for i in range(points_3D.shape[0]):
    pnts = pm.geodetic2enu(points_3D[i,0], points_3D[i,1], points_3D[i,2], POI[0], POI[1], POI[2])
    new_pt3D.append(pnts)
pts3D = np.array(new_pt3D)

if args.displaygraphs:
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts3D[:,0], pts3D[:,1], pts3D[:,2], zdir='z')
    for i in range(pts3D.shape[0]):
        ax.text(pts3D[i,0],pts3D[i,1],pts3D[i,2],  '%s' % (str(i+1)), size=20, zorder=1, color='k')
    ax.set_aspect('equal')
    plt.show()

# Get images from directory
print(f"Searching images in ./sets/C0042/")
images = sorted(glob.glob(f'./sets/C0042/*.jpg'), key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

pbar = tqdm(desc='READING FRAMES', total=len(images), unit=' frames')
if st_frame != 0:
    pbar.update(st_frame)
for fname in images[st_frame:8886]:
    img = cv.imread(fname)
    ffname = fname.split("\\")[-1][:-4]

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = cv.medianBlur(img_gray, 5)
    thr = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, -32)
        
    if fname != images[st_frame]:
        # Detect new position of CODETARGETS
        kp1, des1 = orb.detectAndCompute(img_old,None)
        kp2, des2 = orb.detectAndCompute(img_gray,None)
        
        # Match descriptors.
        matches = bf.match(des1,des2)
        dmatches = sorted(matches, key=lambda x:x.distance)
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)
        
        # img3 = cv.drawMatches(img_old,kp1,img_gray,kp2,matches[::2],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # displayImage(img3, name=fname)
        
        # Find homography matrix and do perspective transform to ct_points
        M, mask = cv.findHomography(src_pts, dst_pts, cv.LMEDS, 5.0)
        pts2D = cv.perspectiveTransform(pts2D_old, M)
        
        # Remove CODETARGETS if reprojections are not inside the image
        pts2D_nn = [pts2D[i] for i in range(pts2D.shape[0]) if pts2D[i,0,0] > 0 and pts2D[i,0,1] > 0]
        points_2D = np.array(pts2D_nn)
                
    # _, rvec, tvec = cv.solvePnP(pts3D, points_2D, cameraMatrix=mtx, distCoeffs=dist_coeff)
    # pts2D_repr = cv.projectPoints(pts3D, rvec, tvec, cameraMatrix=mtx, distCoeffs=dist_coeff)[0]

    # if images.index(fname) % 500 == 0:
    #     displayImageWPoints(img, points_2D)

    pts2D_new = []
    for pt in points_2D:
        npt = pt.reshape(2).astype(int)
        wdw = 25
        contours, _ = cv.findContours(thr[npt[1]-wdw:npt[1]+wdw,npt[0]-wdw:npt[0]+wdw],cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        
        # Calculate moments
        try:
            M = cv.moments(contours[-1])
        except: 
            displayImage(thr[npt[1]-wdw:npt[1]+wdw,npt[0]-wdw:npt[0]+wdw])
        
        # Calculate x,y
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        pts2D_new.append([[npt[0]-wdw+cX, npt[1]-wdw+cY]])
    
    pts2D_new = np.array(pts2D_new, dtype=np.float64)
    _, rvec, tvec = cv.solvePnP(pts3D, pts2D_new, cameraMatrix=mtx, distCoeffs=dist_coeff)
    
    points2D_all.append(pts2D_new)
    vecs.append(np.array([rvec, tvec]))
    ret_names.append(ffname)
    
    if args.save:            
        vid_data = {'2D_points': points2D_all, 'rt_vectors': vecs, 'frame_name': ret_names, 'last_passed_frame': images.index(fname)}
        with open(f'./datasets/dataTOCO_vid1.pkl', 'wb') as fp:
            pickle.dump(vid_data, fp)
    
    pts2D_old = pts2D_new
    img_old = img_gray
    pbar.update(1)
pbar.close()

img = cv.imread(images[0])

if args.save:
    vid_data = {'3D_points': pts3D, '2D_points': points2D_all, 'frame_name': ret_names,
                'init_mtx': mtx, 'init_dist': dist_coeff, 'img_shape': img.shape[1::-1],
                'init_calibfile': args.calibfile, 'rt_vectors': vecs}
    with open(f'./datasets/pkl-files/TOCO_vid1.pkl', 'wb') as fp:
        pickle.dump(vid_data, fp)
        print(f"Dictionary saved successfully as './datasets/pkl-files/TOCO_vid1.pkl'")

if args.displaygraphs:
    displayImageWPoints(img, points_2D, name=ret_names[-1])