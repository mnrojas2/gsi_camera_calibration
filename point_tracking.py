#!/usr/bin/env python

import os
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
from scipy.spatial import distance

# Initialize parser
parser = argparse.ArgumentParser(description='Camera calibration using chessboard images.')
parser.add_argument('folder', type=str, help='Name of the folder containing the frames (*.jpg).')
parser.add_argument( '-e', '--extended', action='store_true', default=False, help='Enables use of cv.calibrateCameraExtended instead of the default function.')
parser.add_argument( '-k', '--k456', action='store_true', default=False, help='Enables use of six radial distortion coefficients instead of the default three.')
parser.add_argument( '-p', '--plot', action='store_true', default=False, help='Shows or saves plots of every frame with image points and projected points.')
parser.add_argument('-cb', '--calibfile', type=str, metavar='file', help='Name of the file containing calibration results (*.yml).')
parser.add_argument('-hf', '--halfway', type=str, metavar='target_data', help='Name of the file containing target data to restart tracking process from any frame (*.txt).')
parser.add_argument( '-s', '--save', action='store_true', default=False, help='Saves calibration data results in .yml format as well as TARGET position data in .txt format.')
parser.add_argument('-o', '--outf', type=str, metavar='out-folder', help='Name of the folder containing frames with their points (--save must be called).')

###############################################################################################################################
# Functions

def displayImage(img, width=1280, height=720, name='Picture'):
    # Small simple function to display images without needing to add the auxiliar functions. By default it reduces the size of the image to 1280x720.
    cv.imshow(name, cv.resize(img, (width, height)))
    cv.waitKey(0)
    cv.destroyAllWindows()
        
def displayImageWPoints(img, *args, name='Image', show_names=False, save=False, fdir='new_set'):
    # Create output folder if it wasn't created yet
    if not os.path.exists('./tests/tracked-sets/'+fdir):
        os.mkdir('./tests/tracked-sets/'+fdir)
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
        clr = [128, 0, 128]
        if len(args) > 1:
            clr[1] += 128
            clr = (np.array(clr) + np.random.randint(-128, 128, size=3)).tolist()
        for i in range(arg.shape[0]):
            cv.circle(img_copy, values[i], 4, clr, -1)
            if show_names and isinstance(arg, pd.DataFrame):
                cv.putText(img_copy, keys[i], values[i], cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    if save:
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

def deleteFarPoints(dataframe, df_projected, limit=60):
    # Checks distance between points in dataframe and df_projected and deletes those from dataframe that surpass a certain upper limit
    new_df = dataframe.reset_index()
    new_dfX = new_df["index"].map(df_projected["X"])
    new_dfY = new_df["index"].map(df_projected["Y"])
    new_dfXY = pd.concat([new_dfX, new_dfY], axis=1)
    
    new_df['dist'] = np.linalg.norm(new_df.iloc[:, [1,2]].values - new_dfXY, axis=1)
    new_df_list = new_df.loc[new_df['dist'] >= limit]['index'].to_list()
    return dataframe.drop(new_df_list)

###############################################################################################################################
# Parameters

# GSI data import
# Import the 3D points from the csv file
obj_3D = pd.read_csv('./videos/Coords/Bundle_fix.csv')[['X','Y','Z']]
points_3D = obj_3D.to_numpy() # BEWARE: to_numpy() doesn't generate a copy but another instance to access the same data. So, if points_3D changes, obj3D will too.

# Point of interest (center)
POI = obj_3D.loc[['CODE45']].to_numpy()[0]
points_3D -= POI

# Crossmatch
# Initialize crossmatching algorithm functions
orb = cv.ORB_create(WTA_K=4, edgeThreshold=255, patchSize=255)
bf = cv.BFMatcher(cv.NORM_HAMMING2, crossCheck=True)

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

# Other parameters
# Minimum number of CODETARGETS necessary for 3D reconstruction
min_corners = 6
mid_val = ['TARGET119', 'TARGET220', 'TARGET201', 'TARGET336', 'TARGET339', 
           'TARGET274', 'TARGET283', 'TARGET316', 'TARGET297', 'TARGET111', 
           'TARGET53', 'TARGET47', 'TARGET242', 'TARGET83', 'TARGET62']
        

###############################################################################################################################
# Main

# Get parser arguments
args = parser.parse_args()
    
if args.halfway:
    # Load .txt file with some specific frame codetarget locations
    with open(f'./tests/points-data/{args.halfway}.txt') as json_file:
        frame_dict = json.load(json_file)
    
    # Save starting point
    start_frame = int(frame_dict['last_passed_frame'][5:])
    frame_dict.pop('last_passed_frame')
    
    # Get point values
    df_frame = pd.DataFrame.from_dict(frame_dict)
    cframes = df_frame.to_numpy().reshape(-1,1,2)

    # Get CODETARGET locations in image
    ct_idx = [df_frame.index.get_loc(idx) for idx in df_frame.index if 'CODE' in idx]
    ct_names = [idx for idx in df_frame.index if 'CODE' in idx]
    ct_corners = cframes[ct_idx]
    
    # Get CODETARGET locations in 3D
    ct_points_3D = obj_3D.loc[ct_names].to_numpy()
else:
    # Load .txt file with some specific frame codetarget locations found manually (usually frame 0)
    with open(f'./tests/points-data/pts-start.txt') as json_file:
        frame_dict = json.load(json_file)
    
    # Use CODETARGET values from ct_frame_dict (default=image0)
    ct_frame_dict = frame_dict[args.folder]
    start_frame = 0
    
    # Get CODETARGET locations in image
    ct_corners = np.array(list(ct_frame_dict.values()), dtype=np.float64)
    
    # Get CODETARGET locations in 3D
    ct_points_3D = obj_3D.loc[ct_frame_dict.keys()].to_numpy()

###############################################################################################################################
# Replace local camera calibration parameters from file (if enabled)
if args.calibfile:
    fs = cv.FileStorage('./tests/results/'+args.calibfile+'.yml', cv.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeff = fs.getNode("dist_coeff").mat()[:8]
    print(f'Imported calibration parameters from /{args.calibfile}.yml/')

###############################################################################################################################
# Initial process

# Get images from directory
print(f"Searching images in ./sets/{args.folder}/")
images = sorted(glob.glob(f'./sets/{args.folder}/*.jpg'), key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

# Arrays to store object points and image points from all frames possible.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.
ret_names = [] # names of every frame for tabulation

###############################################################################################################################
# Image processing

pbar = tqdm(desc='READING FRAMES', total=len(images), unit=' frames')
if start_frame != 0:
    pbar.update(start_frame)
for fname in images[start_frame:]:
    # Read image
    img0 = cv.imread(fname)
    ffname = fname[8+len(args.folder):-4]
    
    # Detect points in image
    img_gray = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    
    # Applying threshold to find points
    thr = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 51, -64)
    
    if int(ffname[5:]) != start_frame:
        # Detect new position of CODETARGETS
        kp1, des1 = orb.detectAndCompute(img_old,None)
        kp2, des2 = orb.detectAndCompute(img_gray,None)
        
        # Match descriptors.
        matches = bf.match(des1,des2)
        dmatches = sorted(matches, key=lambda x:x.distance)
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)
        
        img3 = cv.drawMatches(img_old,kp1,img_gray,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.imshow(img3),plt.show()
        
        # Find homography matrix and do perspective transform to ct_points
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        ct_corners_proy = cv.perspectiveTransform(ct_corners, M)
        
        # Remove CODETARGETS if reprojections are not inside the image
        nn_ct_proy = [ct_corners_proy[i] for i in range(ct_corners_proy.shape[0]) if ct_corners_proy[i,0,0] > 0 and ct_corners_proy[i,0,1] > 0]
        nn_ct_name = [ct_corners_names[i] for i in range(ct_corners_proy.shape[0]) if ct_corners_proy[i,0,0] > 0 and ct_corners_proy[i,0,1] > 0]
                
        ct_corners_proy = np.array(nn_ct_proy)
        ct_corners_names = nn_ct_name
            
        # Find the correct position of points using a small window and getting the highest value closer to the center.
        ct_corners = []
        ct_names_fix = []
        for i in range(ct_corners_proy.shape[0]):
            cnr = ct_corners_proy[i]
            wd = 24
            x_min = 0 if int(cnr[0,0] - wd) <= 0 else int(cnr[0,0] - wd)
            x_max = w if int(cnr[0,0] + wd) >= w else int(cnr[0,0] + wd)
            y_min = 0 if int(cnr[0,1] - wd) <= 0 else int(cnr[0,1] - wd)
            y_max = h if int(cnr[0,1] + wd) >= h else int(cnr[0,1] + wd)
            
            contours, _ = cv.findContours(thr[y_min:y_max, x_min:x_max],cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 3 or (len(contours) > 1 and 'TARGET' in ct_corners_names[i]):
                cntrs = []
                for c in contours:
                    # Calculate moments for each contour
                    M = cv.moments(c)
                    
                    # Calculate x,y coordinate of center
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    cntrs.append([cX, cY])
                cntrs = np.array(cntrs[1:]) # First is dropped since it's always (not sure) pointing the center of the patch
                c_idx = distance.cdist(cntrs, cntrs.mean(axis=0).reshape(1, 2)).argmin()
                cX, cY = cntrs[c_idx]
                ct_corners.append([[x_min+cX, y_min+cY]])
                ct_names_fix.append(ct_corners_names[i])
        
        ct_corners = np.array(ct_corners, dtype=np.float64) # index = ct_corners_names
        ct_corners_names = ct_names_fix
        ct_points_3D = obj_3D.loc[ct_corners_names].to_numpy()
        
    ###########################################################################################################################
    # Find rest of points using CODETARGET projections
    if args.calibfile:
        # Get angle of camera by matching known 2D points with 3D points
        res, rvec, tvec = cv.solvePnP(ct_points_3D, ct_corners, camera_matrix, dist_coeff)
        
        # Make simulated image with 3D points data
        points_2D = cv.projectPoints(points_3D, rvec, tvec, camera_matrix, dist_coeff)[0]
    else:
        # Solve the matching without considering distortion coefficients
        res, rvec, tvec = cv.solvePnP(ct_points_3D, ct_corners, camera_matrix, None)
        points_2D = cv.projectPoints(points_3D, rvec, tvec, camera_matrix, None)[0]
        
    df_points_2D = pd.DataFrame(data=points_2D[:,0,:], index=obj_3D.index.to_list(), columns=['X', 'Y'])
    
    # List position of every point found
    contours, _ = cv.findContours(thr,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    corners = []
    for c in contours:
        # Calculate moments for each contour
        M = cv.moments(c)

        # Calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        corners.append([[cX, cY]])

    # Create a list of corners (equivalent of findCirclesGrid)
    corners = np.array(corners, dtype=np.float32)

    # Get distance between 2D projected points and 2D image points
    corners_matrix = distance.cdist(corners[:,0,:], points_2D[:,0,:])

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
    
    # If number of CODETARGETS is under 6, add TARGETS near the middle of the image to compensate
    if len(ct_corners_names) <= min_corners:
        
        mid_pt = np.array([[w/2, h/2]])
        dist2mid_pt = (distance.cdist(df_corners[['X', 'Y']], mid_pt)).reshape(-1)
        
        # mid_val = []
        # for val in dist2mid_pt.argsort():
        #     if 'CODE' not in df_corners.iloc[val].name and len(mid_val) < (min_corners+3)-len(ct_corners_names):
        #         mid_val.append(df_corners.iloc[val].name)
        
        ct_corners_idx = [df_corners.index.get_loc(idx) for idx in df_corners.index if 'CODE' in idx or idx in mid_val]
        ct_corners_names = [idx for idx in df_corners.index if 'CODE' in idx or idx in mid_val]
    ct_corners = new_corners[ct_corners_idx]
    
    # Save CODETARGETS data in a .txt file in case it's necessary to restart halfway through the process.
    if args.save:
        save_corners = df_corners.to_dict()
        save_corners['last_passed_frame'] = ffname
        with open(f'./tests/points-data/data{args.folder}.txt', 'w') as fp:
            json.dump(save_corners, fp, indent=4)
    
    # Show or save frames with points
    if args.plot:
        displayImageWPoints(img0, df_corners, name=ffname, show_names=True, save=True, fdir=args.outf)
        # df_ct = pd.DataFrame(data=ct_corners[:,0,:], index=ct_corners_names, columns=['X', 'Y'])
        # displayImageWPoints(img0, df_ct, name=ffname, show_names=True)

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