#!/usr/bin/env python

import cv2 as cv
import numpy as np
import pandas as pd
import glob
import argparse
import json
import re
import copy
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance

# Initialize parser
parser = argparse.ArgumentParser(description='Camera calibration using chessboard images.')
parser.add_argument('folder', type=str, help='Name of the folder containing the frames (*.jpg).')
parser.add_argument('-e', '--extended', action='store_true', default=False, help='Enables use of cv.calibrateCameraExtended instead of the default function.')
parser.add_argument('-k', '--k456', action='store_true', default=False, help='Enables use of six radial distortion coefficients instead of the default three.')
parser.add_argument('-th', '--threshold', type=int, metavar='N', default=128, choices=range(256), help='Value of threshold to generate binary image with all but target points filtered.')
parser.add_argument('-p', '--plots', action='store_true', default=False, help='Shows plots of every frame with image points and projected points.')
parser.add_argument('-s', '--save', action='store_true', default=False, help='Saves calibration data results in .yml format.')
parser.add_argument('-cb', '--calibfile', type=str, metavar='file', help='Name of the file containing calibration results (*.yml).')

#############################################################################
# Functions

def displayImage(img, width=1280, height=720, name='Picture'):
    # Small simple function to display images without needing to add the auxiliar functions. By default it reduces the size of the image to 1280x720.
    cv.imshow(name, cv.resize(img, (width, height)))
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def displayImageWArrays(img, *args, name='Image', save=False):
    if img.ndim == 2:
        img_copy = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    else:
        img_copy = copy.copy(img)
    for arg in args:
        clr = [128, 0, 128]
        if len(args) > 1:
            clr[1] += 128
            clr = (np.array(clr) + np.random.randint(-128, 128, size=3)).tolist()
        for i in range(arg.shape[0]):
            cv.circle(img_copy, arg[i].astype(int), 4, clr, -1)
    if save:
        cv.imwrite(f'./tests/fC51e/{name}.jpg', img_copy)
    else:
        displayImage(img_copy, name=name)
        
def displayImageWDataframe(img, *args, name='Image', show_names=False, save=False):
    if img.ndim == 2:
        img_copy = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    else:
        img_copy = copy.copy(img)
    for arg in args:
        keys = arg.index.to_list()
        values = arg.to_numpy().astype(int)
        clr = [128, 0, 128]
        if len(args) > 1:
            clr[1] += 128
            clr = (np.array(clr) + np.random.randint(-128, 128, size=3)).tolist()
        for i in range(arg.shape[0]):
            cv.circle(img_copy, (int(values[i,0]), int(values[i,1])), 4, clr, -1)
            if show_names:
                cv.putText(img_copy, keys[i], values[i], cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    if save:
        cv.imwrite(f'./tests/fC51e/{name}.jpg', img_copy)
    else:
        displayImage(img_copy, name=name)
    
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
# GSI data import

# Import the 3D points from the csv file
obj_3D = pd.read_csv('./videos/Coords/Bundle_fix.csv')[['X','Y','Z']]
points_3D = obj_3D.to_numpy() # BEWARE: to_numpy() doesn't generate a copy but another instance to access the same data. So, if points_3D changes, obj3D will too.

# Point of interest (center)
POI = obj_3D.loc[['CODE45']].to_numpy()[0]
points_3D -= POI

# Manually found (codetargets) list (C51Finf)
ct_frame_dict = {
		"CODE25": [1581, 588],
		"CODE26": [2635, 231],
		"CODE29": [1816, 599],
		"CODE30": [1860, 913],
		"CODE31": [2091, 228],
		"CODE32": [2614, 920],
		"CODE36": [1302, 579],
		"CODE38": [3078, 911],
		"CODE42": [1293, 914],
		"CODE43": [3105, 229],
		"CODE45": [2089, 910],
		"CODE46": [1574, 909],
		"CODE133": [1402, 957],
		"CODE134": [1789, 960]
	}
ct_frame = np.array(list(ct_frame_dict.values()), dtype=np.float64)
ct_points_3D = obj_3D.loc[ct_frame_dict.keys()].to_numpy()

#############################################################################
# Crossmatch

# Initialize crossmatching algorithm functions
orb = cv.ORB_create(WTA_K=4, edgeThreshold=255, patchSize=255)
bf = cv.BFMatcher(cv.NORM_HAMMING2, crossCheck=True)

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

dist_coeff = np.array(([k1], [k2], [p1], [p2], [k3]))

################################################################################
# Main
args = parser.parse_args()

# Replace local camera calibration parameters from file (if enabled)
if args.calibfile:
    fs = cv.FileStorage('./tests/'+args.calibfile+'.yml', cv.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeff = fs.getNode("dist_coeff").mat()[:8]
    print(f'Imported calibration parameters from /{args.calibfile}.yml/')

    # Get angle of camera by matching known 2D points with 3D points
    res, rvec, tvec = cv.solvePnP(ct_points_3D, ct_frame, camera_matrix, dist_coeff)
    
    # Make simulated image with 3D points data
    points_2D = cv.projectPoints(points_3D, rvec, tvec, camera_matrix, dist_coeff)[0]
else:
    # Solve the matching without considering distortion coefficients
    res, rvec, tvec = cv.solvePnP(ct_points_3D, ct_frame, camera_matrix, None)
    points_2D = cv.projectPoints(points_3D, rvec, tvec, camera_matrix, None)[0]
    
df_points_2D = pd.DataFrame(data=points_2D[:,0,:], index=obj_3D.index.to_list(), columns=['X', 'Y'])

# Get images from directory
print(f"Searching images in ./sets/{args.folder}/")
images = sorted(glob.glob(f'./sets/{args.folder}/*.jpg'), key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

# Arrays to store object points and image points from all frames possible.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.
ret_names = [] # names of every frame for tabulation

######################################################################################################################################
# Image 0 name
fname = images[0]
    
# Read image
img0 = cv.imread(fname)
ffname = fname[8+len(args.folder):-4]

# Detect points in image
img_gray = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)

# Applying threshold to find points
thr = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 51, -128)

# List position of every point found
contours, hierarchy = cv.findContours(thr,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
corners = []
for c in contours:
    # calculate moments for each contour
    M = cv.moments(c)

    # calculate x,y coordinate of center
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    corners.append([[cX, cY]])

# Create a list of corners (equivalent of findCirclesGrid)
corners = np.array(corners, dtype=np.float32)

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
new_corners = df_cnp.reshape(-1,1,2)
new_obj3D = obj_3D.loc[df_corners.index.to_list()].to_numpy(dtype=np.float32)

# Get position of CODETARGETS
ct_corners_idx = [df_corners.index.get_loc(idx) for idx in df_corners.index if 'CODE' in idx]
ct_corners_names = [idx for idx in df_corners.index if 'CODE' in idx]
ct_corners = new_corners[ct_corners_idx]

# Save 3D and 2D point data for calibration
objpoints.append(new_obj3D)
imgpoints.append(new_corners)
ret_names.append(ffname)

img_old = img_gray

# Rest of images
pbar = tqdm(desc='READING FRAMES', total=len(images)-1, unit=' frames')
for fname in images[1:]:
    # Read image
    img0 = cv.imread(fname)
    ffname = fname[8+len(args.folder):-4]
    
    # Detect points in image
    img_gray = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
    
    # Applying threshold to find points
    thr = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 51, -128)
    
    ###########################################################################################################################
    # Detect new position of CODETARGETS
    kp1, des1 = orb.detectAndCompute(img_old,None)
    kp2, des2 = orb.detectAndCompute(img_gray,None)
    
    # img1 = cv.drawKeypoints(img_old, kp1, None, color=(0,255,0), flags=0)
    # img2 = cv.drawKeypoints(img_gray, kp2, None, color=(0,255,0), flags=0)
    # plt.figure(), plt.imshow(img1), plt.figure(), plt.imshow(img2), plt.show()
    
    # Match descriptors.
    matches = bf.match(des1,des2)
    dmatches = sorted(matches, key=lambda x:x.distance)
    
    # img3 = cv.drawMatches(img_old,kp1,img_gray,kp2,dmatches[::4],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.figure(), plt.imshow(img3), plt.show()
    # cv.imwrite(f'./tests/fC51/frames({int(ffname_old[5:])}-{int(ffname[5:])}).jpg', img3)
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)
    
    # Find homography matrix and do perspective transform to ct_points
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    # M, mask = cv.findEssentialMat(src_pts, dst_pts, cameraMatrix=camera_matrix, method=cv.RANSAC, prob=0.9, threshold=1.0)
    
    ct_corners_proy = cv.perspectiveTransform(ct_corners, M)
    
    if int(ffname[5:]) >= 960:
        # print('after reprojecting codetargets: \n', ct_corners_proy[:,0,:], '\n', ct_corners_names)
        print(f'length projected {ct_corners_proy.shape}, length names: {len(ct_corners_names)}')
    
    # Remove CODETARGETS if reprojections are not inside the image
    nn_ct_corners_proy = []
    nn_ct_corners_name = []
    for i in range(ct_corners_proy.shape[0]):
        cnr = ct_corners_proy[i]
        cnr_n = ct_corners_names[i]
        if cnr[0,0] > 0 and cnr[0,1] > 0:
            nn_ct_corners_proy.append(cnr)
            nn_ct_corners_name.append(cnr_n)
            
    ct_corners_proy = np.array(nn_ct_corners_proy)
    ct_corners_names = nn_ct_corners_name
        
    # Find the correct position of points using a small window and getting the highest value closer to the center.
    h, w = img_gray.shape
    ct_corners = []
    for i in range(ct_corners_proy.shape[0]):
        cnr = ct_corners_proy[i]
        wd = 24
        x_min = 0 if int(cnr[0,0] - wd) <= 0 else int(cnr[0,0] - wd)
        x_max = w if int(cnr[0,0] + wd) >= w else int(cnr[0,0] + wd)
        y_min = 0 if int(cnr[0,1] - wd) <= 0 else int(cnr[0,1] - wd)
        y_max = h if int(cnr[0,1] + wd) >= h else int(cnr[0,1] + wd)
        
        contours, hierarchy = cv.findContours(thr[y_min:y_max, x_min:x_max],cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        
        # Después del frame966 no hay más valores en el vector de codetargets
        if len(contours) > 1:
            cntrs = []
            for c in contours:
                # calculate moments for each contour
                M = cv.moments(c)

                # calculate x,y coordinate of center
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                cntrs.append([cX, cY])
            cntrs = np.array(cntrs[1:]) # First is dropped since it's always (not sure) pointing the center of the patch
            c_idx = distance.cdist(cntrs, cntrs.mean(axis=0).reshape(1, 2)).argmin()
            cX, cY = cntrs[c_idx]
            ct_corners.append([[x_min+cX, y_min+cY]])
        else:
            del ct_corners_names[i]
    
    ct_corners = np.array(ct_corners, dtype=np.float64) # index = ct_corners_names
    ct_points_3D = obj_3D.loc[ct_corners_names].to_numpy()
    
    #########################################################################################
    # Find rest of points using CODETARGET projections
    
    if args.calibfile:
        fs = cv.FileStorage('./tests/'+args.calibfile+'.yml', cv.FILE_STORAGE_READ)
        camera_matrix = fs.getNode("camera_matrix").mat()
        dist_coeff = fs.getNode("dist_coeff").mat()[:8]
        print(f'Imported calibration parameters from /{args.calibfile}.yml/')

        # Get angle of camera by matching known 2D points with 3D points
        res, rvec, tvec = cv.solvePnP(ct_points_3D, ct_corners, camera_matrix, dist_coeff)
        
        # Make simulated image with 3D points data
        points_2D = cv.projectPoints(points_3D, rvec, tvec, camera_matrix, dist_coeff)[0]
    else:
        # Solve the matching without considering distortion coefficients
        res, rvec, tvec = cv.solvePnP(ct_points_3D, ct_corners, camera_matrix, None)
        points_2D = cv.projectPoints(points_3D, rvec, tvec, camera_matrix, None)[0]
        
    df_points_2D = pd.DataFrame(data=points_2D[:,0,:], index=obj_3D.index.to_list(), columns=['X', 'Y'])
    
    # df_ct_corn = pd.DataFrame(data=ct_corners[:,0,:], index=ct_corners_names, columns=['X', 'Y'])
    # displayImageWDataframe(img0, df_ct_corn, name=ffname, show_name=True, save=True)
    
    # List position of every point found
    contours, hierarchy = cv.findContours(thr,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    corners = []
    for c in contours:
        # calculate moments for each contour
        M = cv.moments(c)

        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        corners.append([[cX, cY]])

    # Create a list of corners (equivalent of findCirclesGrid)
    corners = np.array(corners, dtype=np.float32)

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
    new_corners = df_cnp.reshape(-1,1,2)
    new_obj3D = obj_3D.loc[df_corners.index.to_list()].to_numpy(dtype=np.float32)

    # Get position of CODETARGETS
    ct_corners_idx = [df_corners.index.get_loc(idx) for idx in df_corners.index if 'CODE' in idx]
    ct_corners_names = [idx for idx in df_corners.index if 'CODE' in idx]
    ct_corners = new_corners[ct_corners_idx]
    
    ct_dict_backup = {}
    for i in range(len(ct_corners_names)):
        ct_dict_backup[ct_corners_names[i]] = ct_corners[i].tolist()
    ct_dict_backup['last_passed_frame'] = ffname
        
    with open(f'./tests/data.txt', 'w') as fp:
        json.dump(ct_dict_backup, fp, indent=4)
    
    # displayImageWArrays(img0, new_corners[:,0,:], name=ffname, save=False)
    # displayImageWDataframe(img0, df_corners, name=ffname, show_names=True, save=True)

    # Save 3D and 2D point data for calibration
    objpoints.append(new_obj3D)
    imgpoints.append(new_corners)
    ret_names.append(ffname)

    img_old = img_gray
    pbar.update(1)
pbar.close()

# When everything done, release the frames
cv.destroyAllWindows()

print("We finished!")

# Chequear en la primera parte si los puntos todavía conservan su posición respecto a los datos del obj3D. R: Lo están, pero el orden de los puntos cambió, no es problema ya que los ptos 3D también cambiaron