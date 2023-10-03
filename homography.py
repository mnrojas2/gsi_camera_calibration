#!/usr/bin/env python

import cv2
import numpy as np
import pandas as pd
import glob
import argparse
import json
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance

# Initialize parser
parser = argparse.ArgumentParser(description='Camera calibration using chessboard images.')
parser.add_argument('folder', type=str, help='Name of the folder containing the frames (*.jpg).')
parser.add_argument('-e', '--extended', action='store_true', default=False, help='Enables use of cv2.calibrateCameraExtended instead of the default function.')
parser.add_argument('-k', '--k456', action='store_true', default=False, help='Enables use of six radial distortion coefficients instead of the default three.')
parser.add_argument('-th', '--threshold', type=int, metavar='N', default=128, choices=range(256), help='Value of threshold to generate binary image with all but target points filtered.')
parser.add_argument('-p', '--plots', action='store_true', default=False, help='Shows plots of every frame with image points and projected points.')
parser.add_argument('-s', '--save', action='store_true', default=False, help='Saves calibration data results in .yml format.')
parser.add_argument('-cb', '--calibfile', type=str, metavar='file', help='Name of the file containing calibration results (*.yml).')

#############################################################################
# Functions

def displayImage(img, width=1280, height=720, name='Picture'):
    # Small simple function to display images without needing to add the auxiliar functions. By default it reduces the size of the image to 1280x720.
    cv2.imshow(name, cv2.resize(img, (width, height)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def displayImageWPoints(img, *args, name='Picture'):
    for arg in args:
        for i in range(arg.shape[0]):
            cv2.circle(img, (int(arg[i,0]), int(arg[i,1])), 5, (128, 0, 128), -1)
    displayImage(img, name=name)
    
def scatterPlot(*args, name='Picture'):
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    for arg in args:
        ax.scatter(arg[:,0], -arg[:,1])
    plt.get_current_fig_manager().set_window_title(name)
    plt.show()
    
def deleteDuplicatesPoints(dataframe, df_projected):
    # Takes a list of points with defined targets and deletes those that repeat target by keeping the closest point
    dup_corners = dataframe[dataframe.index.duplicated(keep=False)].reset_index()
    dataframe = dataframe[~dataframe.index.duplicated(keep=False)]

    dup_cornersX = dup_corners["index"].map(df_projected["X"])
    dup_cornersY = dup_corners["index"].map(df_projected["Y"])
    dup_cornersXY = pd.concat([dup_cornersX, dup_cornersY], axis=1)

    dup_corners['dist'] = np.linalg.norm(dup_corners.iloc[:, [1,2]].values - dup_cornersXY, axis=1)
    dup_corners_fix = dup_corners.loc[dup_corners.groupby('index', sort=False)['dist'].idxmin()][['index', 'X', 'Y']]
    dup_corners_fix.set_index('index', inplace=True)
    dup_corners_fix.index.name=None

    dataframe = pd.concat([dataframe, dup_corners_fix], axis=0)
    dataframe = deleteFarPoints(dataframe, df_projected)
    return dataframe

def deleteFarPoints(dataframe, df_projected, limit=75):
    # Checks distance between points in dataframe and df_projected and deletes those from dataframe that surpass a certain upper limit
    new_df = dataframe.reset_index()
    new_dfX = new_df["index"].map(df_projected["X"])
    new_dfY = new_df["index"].map(df_projected["Y"])
    new_dfXY = pd.concat([new_dfX, new_dfY], axis=1)
    
    new_df['dist'] = np.linalg.norm(new_df.iloc[:, [1,2]].values - new_dfXY, axis=1)
    new_df_list = new_df.loc[new_df['dist'] >= limit]['index'].to_list()
    return dataframe.drop(new_df_list)

#############################################################################
# Blob detection parameters

# Termination criteria for cv2.cornerSubPix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 90, 0.00001)

# Initializing parameter setting using cv2.SimpleBlobDetector function
blobParams = cv2.SimpleBlobDetector_Params()

# Filter by Area
blobParams.filterByArea = True
blobParams.maxArea = 175

# Filter by Circularity
blobParams.filterByCircularity = True
blobParams.minCircularity = 0.7

# Creating a blob detector using the defined parameters
blobDetector = cv2.SimpleBlobDetector_create(blobParams)
#############################################################################
# GSI data import

# Import the 3D points from the csv file
obj_3D = pd.read_csv('./videos/Coords/Bundle.csv')[['X','Y','Z']]
points_3D = obj_3D.to_numpy() # BEWARE to_numpy() doesn't generate a copy but another instance to access the same data. So, if points_3D changes, obj3D will too.

# Point of interest (center)
POI = obj_3D.loc[['CODE45']].to_numpy()[0]
points_3D -= POI

# Import manually found points (codetargets) list
points_dir = './tests/points_json.txt'
with open(points_dir) as json_file:
    codetargets = json.load(json_file)

#############################################################################
# Camera intrinsic parameters for calibration

# Camera matrix
fx = 2605.170124
fy = 2596.136808
cx = 1920 # 1882.683683
cy = 1080 # 1072.920820

camera_matrix = np.array([[fx, 0., cx],
                          [0., fy, cy],
                          [0., 0., 1.]], dtype = "double")

# Distortion coefficients
k1 = -0.011935952
k2 =  0.03064728
p1 = -0.00067055
p2 = -0.00512621
k3 = -0.11974069

dist_coeff = np.array(([k1], [k2], [p1], [p2], [k3])) # , [k4], [k5], [k6]))

# Flags
flags_model = cv2.CALIB_USE_INTRINSIC_GUESS
# CALIB_USE_INTRINSIC_GUESS: Calibration needs a preliminar camera_matrix to start (necessary in non-planar cases)
# CALIB_RATIONAL_MODEL: Enable 6 rotation distortion constants instead of 3

################################################################################
# Main
args = parser.parse_args()

# Enable use of calibrateCameraExtended
if args.extended:
    print(f'calibrateCameraExtended function set')

# Replace local camera calibration parameters from file (if enabled)
if args.calibfile:
    fs = cv2.FileStorage('./tests/'+args.calibfile+'.yml', cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeff = fs.getNode("dist_coeff").mat()[:8]
    print(f'Imported calibration parameters from /{args.calibfile}.yml/')

# Camera Calibration Flags
if args.k456:
    flags_model |= cv2.CALIB_RATIONAL_MODEL
    print(f'CALIB_RATIONAL_MODEL flag set')

# Get images from directory
images = glob.glob(f'./tests/{args.folder}/*.jpg')
print(f"Searching images in ./tests/{args.folder}/")

# Arrays to store object points and image points from all frames possible.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.
ret_names = [] # names of every frame for tabulation

pbar = tqdm(desc='READING FRAMES', total=len(images), unit=' frames')
for fname in images:
    # Read image
    img0 = cv2.imread(fname)
    ffname = fname[8+len(args.folder)+1:-4]
    
    # Get list of codetargets from manually found point list
    ct_frame_dict = codetargets[ffname]
    ct_frame = np.array(list(ct_frame_dict.values()), dtype=np.float64)
    ct_points_3D = obj_3D.loc[ct_frame_dict.keys()].to_numpy() # POI doesn't need to get subtracted, already done in line 90, explained in line 86.
    
    if args.calibfile:
        # Get angle of camera by matching known 2D points with 3D points
        res, rvec, tvec = cv2.solvePnP(ct_points_3D, ct_frame, camera_matrix, dist_coeff)
        
        # Make simulated image with 3D points data
        points_2D = cv2.projectPoints(points_3D, rvec, tvec, camera_matrix, dist_coeff)[0]
    else:
        # Solve the matching without considering distortion coefficients
        res, rvec, tvec = cv2.solvePnP(ct_points_3D, ct_frame, camera_matrix, None)
        points_2D = cv2.projectPoints(points_3D, rvec, tvec, camera_matrix, None)[0]
    # r0 = R.from_rotvec(rvec.flatten())
    # print(ffname, r0.as_euler('XYZ', degrees=True))
    
    df_points_2D = pd.DataFrame(data=points_2D[:,0,:], index=obj_3D.index.to_list(), columns=['X', 'Y'])

    # Detect points in image
    img_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

    # Applying threshold to find points
    thr = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, -128)
    keypoints = blobDetector.detect(thr)

    # Create a list of corners (equivalent of findCirclesGrid)
    corners = [[[key.pt[0], key.pt[1]]] for key in keypoints]
    corners = np.array(corners, dtype=np.float32)
    
    if corners.shape != (0,):
        im_with_keypoints = cv2.drawKeypoints(img0, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        im_with_keypoints_gray = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)

        # Get distance between 2D projected points and 2D image points
        corners_matrix = distance.cdist(corners[:,0,:], points_2D[:,0,:]) # <-------- this is the function we need to update to find the correct points

        # Convert matrix array in dataframe with proper index and apply idxmin function to find the name of the closest point (obj_3D.index ~ points_2D)
        corners_dataframe = pd.DataFrame(data=corners_matrix, index=np.arange(0, len(corners), 1), columns=obj_3D.index.to_list())
        corners_min = corners_dataframe.idxmin(axis='columns')

        # Delete duplicate points that were not in the GSI point list
        df_corners = pd.DataFrame(data=corners[:,0,:], index=corners_min.tolist(), columns=['X', 'Y'])
        df_corners = deleteDuplicatesPoints(df_corners, df_points_2D)
        df_cnp = df_corners.to_numpy(dtype=np.float32)

        # Produce datasets and add them to list
        new_corners = df_cnp.reshape(df_cnp.shape[0],1,df_cnp.shape[1])
        new_obj3D = obj_3D.loc[df_corners.index.to_list()].to_numpy(dtype=np.float32)
        
        # Refine the corner locations
        corners2 = cv2.cornerSubPix(im_with_keypoints_gray, new_corners, (11,11), (-1,-1), criteria)
        if args.plots:
            scatterPlot(points_2D[:,0,:], new_corners[:,0,:], corners2[:,0,:], name=ffname)

        objpoints.append(new_obj3D)
        imgpoints.append(corners2)
        ret_names.append(ffname)
    pbar.update(1)
pbar.close()

# When everything done, release the video
cv2.destroyAllWindows()

# Camera Calibration
print("Calculating camera matrix...")
if args.extended:
    ret, mtx, dist, rvecs, tvecs, stdInt, stdExt, pVE = cv2.calibrateCameraExtended(objpoints, imgpoints, img0.shape[1::-1], cameraMatrix=camera_matrix, distCoeffs=dist_coeff, flags=flags_model)
    pVE_extended = np.array((np.array(ret_names, dtype=object), pVE[:,0])).T
    pVE_extended = pVE_extended[pVE_extended[:,1].argsort()]
else:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img0.shape[1::-1], cameraMatrix=camera_matrix, distCoeffs=dist_coeff, flags=flags_model)

print('Camera matrix:\n', mtx)
print('Distortion coefficients:\n', dist)
if args.extended:
    print('Error per frame:\n', pVE_extended)

if args.save:
    fs = cv2.FileStorage('./tests/res-'+args.folder+'.yml', cv2.FILE_STORAGE_WRITE)
    summary = input("Insert comments: ")
    fs.write('summary', summary)
    fs.write('init_cam_calib', args.calibfile)
    fs.write('camera_matrix', mtx)
    fs.write('dist_coeff', dist)
    if args.extended:
        pVElist = np.array((np.array([int(x[5:]) for x in ret_names]), pVE[:,0])).T
        fs.write('std_intrinsics', stdInt)
        fs.write('std_extrinsics', stdExt)
        fs.write('per_view_errors', pVElist)
    fs.release()