#!/usr/bin/env python

import os
import re
import copy
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
import cv2 as cv
import pymap3d as pm
import skimage.color as skc
from tqdm import tqdm
from matplotlib import pyplot as plt
from photutils.centroids import centroid_com


# Initialize parser
parser = argparse.ArgumentParser(description='Finds target points in images and correlate them with 3D target point data.')
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
        if not os.path.exists('./sets/tracked-sets/'+fdir):
            os.mkdir('./sets/tracked-sets/'+fdir)
        cv.imwrite(f'./sets/tracked-sets/{fdir}/{name}.jpg', img_copy)
    else:
        displayImage(img_copy, name=name)
    
def scatterPlot(*args, name='Image'):
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    for arg in args:
        ax.scatter(arg[:,0], -arg[:,1])
    plt.get_current_fig_manager().set_window_title(name)
    plt.show()
    

def delta_E(image_1_rgb, color_target, sigma, dmax, p2Dn):
    # color filter
    Lab1 = cv.cvtColor((image_1_rgb/255).astype(np.float32), cv.COLOR_BGR2LAB)
    
    # color_target = np.array([79.05, 142.29, 206.81])/255 # 79.05, 142.29, 206.81
    Lab2 = skc.rgb2lab(color_target.reshape(1, 1, 3))

    deltae1 = skc.deltaE_ciede2000(Lab1, Lab2)
    deltae = cv.GaussianBlur(deltae1, (0,0), 3)

    if p2Dn == 'pt8':
        displayImage(cv.normalize(deltae, None, 0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_32F))

    minDeltaE = np.min(deltae)
    
    if minDeltaE < 100: # maximum threshold to say there's a target here
        fimage = dmax*np.exp(-(deltae-minDeltaE)**2/(2*(sigma**2))) # sigma=2 generally used in color space
        fimage[fimage<0.65] = 0
    else:
        fimage = 0*np.ones_like(deltae) #np.nan
    
    return fimage, minDeltaE

################################################################
# Main

# Get parser arguments
args = parser.parse_args()

toco = 1

################################################################
# GSI data import
# Import the 3D points from the csv file
gpsArr = pd.read_csv(f'./TOCO/datasets/April22GPS/{args.file}.txt', header=None, names=['Lat','Lon','Alt', 'null'])
gpsArr.dropna(axis=1, inplace=True)

points_3D = gpsArr.to_numpy() # BEWARE: to_numpy() doesn't generate a copy but another instance to access the same data. So, if points_3D changes, obj3D will too.

# Point of interest (center)
POI = gpsArr.loc[['pt5']].to_numpy()[0]

# Convert GPS data to cartesian (local based)
pts3D = []
for i in range(points_3D.shape[0]):
    pnts = pm.geodetic2enu(points_3D[i,0], points_3D[i,1], points_3D[i,2], POI[0], POI[1], POI[2])
    pts3D.append(pnts)
df_pts3D = pd.DataFrame(data=np.array(pts3D, dtype=np.float64), index=gpsArr.index.to_list(), columns=['X', 'Y', 'Z'])

points_3D = df_pts3D.to_numpy()
points_3D_names = df_pts3D.index.to_list()

################################################################
# Crossmatch
# Initialize crossmatching algorithm functions
orb = cv.ORB_create()
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

target_color = np.array([0.310, 0.558, 0.811])

# Load .yml file
if args.calibfile:
    print(f'Loading {args.calibfile}.yml')
    fs = cv.FileStorage(f'./results/{args.calibfile}.yml', cv.FILE_STORAGE_READ)

    # Camera intrinsic parameters for calibrations
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeff = fs.getNode("dist_coeff").mat()
    summary = fs.getNode("summary").string()
    
    print(f"File '{args.calibfile}.yml' description:", summary)
    
else:
    # Use averaged values
    # Camera matrix
    fx = 2569.861833844866
    fy = 2568.651262555803
    cx = 1881.5969273158478
    cy = 1087.1150791713173

    camera_matrix = np.array([[fx, 0., cx],
                            [0., fy, cy],
                            [0., 0., 1.]], dtype = "double")

    # Distortion coefficients
    k1 = 0.019007359983931695
    k2 = -0.04099061331944541
    p1 = -0.00032026402966169244
    p2 = -0.0011026758947991768
    k3 = 0.02882744722467448

    dist_coeff = np.array(([k1], [k2], [p1], [p2], [k3]))

################################################################
# Points 2D
frame_dict = {
    "TOCOvid0": {
        'pt1': [2171,  220],
        'pt3': [2191,  884],
        'pt4': [2071, 1303],
        'pt5': [1833,  750],
        'pt6': [1937, 1444],
        'pt7': [1714, 1293],
        'pt8': [1475, 1784],
        'pt9': [1184, 1081],
        'pt10': [1048, 1599],
        'pt11': [1103,  315]
    },
    "TOCOvid1": {
        'pt1': [2188,  281],
        'pt3': [2211,  941],
        'pt4': [2091, 1361],
        'pt5': [1850,  808],
        'pt6': [1955, 1501],
        'pt7': [1731, 1350],
        'pt8': [1488, 1844],
        'pt9': [1197, 1139],
        'pt10': [1061, 1657],
        'pt11': [1121,  383]
    }
}

# Get point values
df_frame = pd.DataFrame.from_dict(frame_dict[f"TOCOvid{toco}"], orient='index')
points_2D = df_frame.to_numpy().astype(np.float64).reshape(-1,1,2) # Note: solvePnP will fail if values in points2D aren't in float format.
points_2D_names = df_frame.index.to_list()

################################################################

st_frame = 0
points2D_all = []
ret_names = []
vecs = []

if args.halfway:
    # Load .txt file with some specific frame codetarget locations
    print(f'./TOCO/datasets/dataTOCO_vid-{toco}.pkl')
    pFile = pickle.load(open(f"./datasets/dataTOCO_vid-{toco}.pkl","rb"))
    
    points2D_all = pFile['2D_points']
    ret_names = pFile['frame_name']
    vecs = pFile['rt_vectors']
    
    points_2D = np.array(points2D_all[-1])
    
    # Save starting point
    st_frame = 1+int(pFile['last_passed_frame'])

if args.displaygraphs:
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3D[:,0], points_3D[:,1], points_3D[:,2], zdir='z')
    for i in range(points_3D.shape[0]):
        ax.text(points_3D[i,0], points_3D[i,1], points_3D[i,2],  '%s' % (str(i+1)), size=20, zorder=1, color='k')
    ax.set_aspect('equal')
    plt.show()

# Get images from directory
print(f"Searching images in ./sets/C004{2*toco}/")
images = sorted(glob.glob(f'./sets/C004{2*toco}/*.jpg'), key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

##############
C = []
##############

pbar = tqdm(desc='READING FRAMES', total=len(images), unit=' frames', dynamic_ncols=True)
if st_frame != 0:
    pbar.update(st_frame)
for fname in images[st_frame:]:
    img = cv.imread(fname)
    ffname = fname.split("\\")[-1][:-4]

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = cv.medianBlur(img_gray, 5) # cv.GaussianBlur(img_gray, (5,5), 0) #
    # thr = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, -32)
        
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
        pts2D_names = [points_2D_names[i] for i in range(pts2D.shape[0]) if pts2D[i,0,0] > 0 and pts2D[i,0,1] > 0]
        points_2D = np.array(pts2D_nn, dtype=np.float64)
        points_2D_names = pts2D_names

        # Fix position if points are slightly shifted
        pts2D_new = []
        pts2D_names = []
        for i in range(points_2D.shape[0]):
            npt = points_2D[i].reshape(2).astype(int)
            wdw = 15
                
            img_patch = img[npt[1]-wdw:npt[1]+wdw,npt[0]-wdw:npt[0]+wdw]
            mask_box, _ = delta_E(img_patch, target_color, sigma=2, dmax=1, p2Dn=points_2D_names[i])
            cXY = centroid_com(mask_box)
            
            # if points_2D_names[i] == 'pt8':
            #     displayImage(cv.normalize(mask_box, None, 0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_32F))
        
            pts2D_new.append([[npt[0]-wdw+cXY[0], npt[1]-wdw+cXY[1]]])
            pts2D_names.append(points_2D_names[i])
        
        pts2D_new = np.array(pts2D_new, dtype=np.float64)
        pts3D_img = df_pts3D.loc[pts2D_names].to_numpy()
        
    else:
        pts2D_new = points_2D
        pts3D_img = df_pts3D.loc[points_2D_names].to_numpy()
    
    # Some 2D points could be lost after passing the filter, so the image is reconstructed from the other points to estimate the location
    _, rvec, tvec = cv.solvePnP(pts3D_img, pts2D_new, cameraMatrix=camera_matrix, distCoeffs=dist_coeff)
    '''pts2D_new2 = cv.projectPoints(df_pts3D.to_numpy(), rvec, tvec, cameraMatrix=camera_matrix, distCoeffs=dist_coeff)[0]
    
    pts2D_nn2 = [pts2D_new2[i] for i in range(pts2D_new2.shape[0]) if pts2D_new2[i,0,0] > 0 and pts2D_new2[i,0,1] > 0]
    pts2D_names = [df_pts3D.index.to_list()[i] for i in range(pts2D_new2.shape[0]) if pts2D_new2[i,0,0] > 0 and pts2D_new2[i,0,1] > 0]
    pts2D_nn2 = np.array(pts2D_nn2, dtype=np.float64) # '''
    
    points2D_all.append(pts2D_new)
    vecs.append(np.array([rvec, tvec]))
    ret_names.append(ffname)
    
    if args.displaygraphs and images.index(fname) % 10 == 0:
        displayImageWPoints(img, points_2D, name=ret_names[-1])
    
    if args.save:            
        vid_data = {'2D_points': points2D_all, 'rt_vectors': vecs, 'frame_name': ret_names, 'last_passed_frame': images.index(fname)}
        with open(f'./TOCO/datasets/dataTOCO_vid0.pkl', 'wb') as fp:
            pickle.dump(vid_data, fp)
    
    pts2D_old = pts2D_new
    img_old = img_gray
    pbar.update(1)
pbar.close()

img = cv.imread(images[0])

if args.save:
    vid_data = {'3D_points': points_3D, '2D_points': points2D_all, 'frame_name': ret_names,
                'init_mtx': camera_matrix, 'init_dist': dist_coeff, 'img_shape': img.shape[1::-1],
                'init_calibfile': args.calibfile, 'rt_vectors': vecs}
    with open(f'./TOCO/datasets/pkl-files/TOCO_vid0.pkl', 'wb') as fp:
        pickle.dump(vid_data, fp)
        print(f"Dictionary saved successfully as './TOCO/datasets/pkl-files/TOCO_vid{toco}.pkl'")

if args.displaygraphs:
    displayImageWPoints(img, points_2D, name=ret_names[-1])