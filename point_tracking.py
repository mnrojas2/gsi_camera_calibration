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
import pickle
import random
import datetime
import camera
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import patches
from scipy.spatial import distance
from scipy.spatial.transform import Rotation as R

# Initialize parser
parser = argparse.ArgumentParser(description='Camera calibration using GSI-based board images. It allows to find targets, save data and/or do the calibration process. Saved data can be used with "just_calibration.py".')
parser.add_argument('folder', type=str, help='Name of the folder containing the frames (*.jpg).')
parser.add_argument('data_3d', type=str, help='Name of the file containing the 3D position of TARGETS and CODETARGETS (*.csv, cartesian units).')
parser.add_argument('data_2d', type=str, help='Name of the file containing the 2D position of CODETARGETS of the first frame to analyze. (*.txt, (x,y) pixel units).')
# Point tracking settings
parser.add_argument('-gm', '--gamma', type=float, default=False, help='Gamma correction')
parser.add_argument('-cb', '--calibfile', type=str, metavar='file', help='Name of the file containing calibration results (*.txt), for point reprojection and/or initial guess during calibration.')
parser.add_argument( '-p', '--plot', action='store_true', default=False, help='Shows or saves plots of every frame with image points and projected points.')
parser.add_argument('-hf', '--halfway', type=str, metavar='target_data', help='Name of the file containing target data to restart tracking process from any frame (*.txt).')
parser.add_argument( '-s', '--save', action='store_true', default=False, help='Saves TARGET position data in .txt format as well as vectors of the 3D and 2D points for calibration.')
# Camera calibration settings
parser.add_argument( '-c', '--calibenable', action='store_true', default=False, help='Enables calibration process after finding all points from video.')
parser.add_argument('-cs', '--calibsave', action='store_true', default=False, help='Saves calibration data results in .yml format.')
# Distance-based filters
parser.add_argument('-fd', '--filterdist', action='store_true', default=False, help='Enables filter by distance of camera position.')
parser.add_argument('-md', '--mindist', type=float, metavar='N', default=0.0, help='Minimum distance between cameras (available only when --filterdist is active).')
parser.add_argument('-ds', '--distshift', type=float, metavar='N', default=0.0, help='Initial shift in distance between cameras (available only when --filterdist is active).')
# Time-based filters
parser.add_argument('-ft', '--filtertime', action='store_true', default=False, help='Enables filter by time between frames.')
parser.add_argument('-rd', '--reduction', type=int, metavar='N', default=1, help='Reduction of number of frames (total/N) (available only when --filtertime is active).')
parser.add_argument('-rs', '--residue', type=int, metavar='N', default=0, help='Residue or offset for the reduced number of frames (available only when --filtertime is active).')
# Point-based filters
parser.add_argument('-fp', '--filterpnts', action='store_true', default=False, help='Enables filter by randomly selecting points on each frame.')
parser.add_argument('-sp', '--split', type=int, metavar='N', default=5, help='Number of sublists generated by splitting the original list in equal parts (available only when --filterpnts is active).')
parser.add_argument('-sf', '--shift', type=int, metavar='N', default=0, help='Selects one specific sublist of the ones created (available only when --filterpnts is active).')

###############################################################################################################################
# Functions

random.seed(0)

def displayImage(img, width=1280, height=720, name='Picture'):
    # Small simple function to display images without needing to add the auxiliar functions. By default it reduces the size of the image to 1280x720.
    cv.imshow(name, cv.resize(img, (width, height)))
    cv.waitKey(0)
    cv.destroyAllWindows()
        
def displayImageWPoints(img, *args, name='Image', show_names=False, save=False, fdir='new_set'):
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
        # Create output folder if it wasn't created yet
        if not os.path.exists('./sets/tracked_sets/'):
            os.mkdir('./sets/tracked_sets/')
        if not os.path.exists('./sets/tracked_sets/'+fdir):
            os.mkdir('./sets/tracked_sets/'+fdir)
        cv.imwrite(f'./sets/tracked_sets/{fdir}/{name}.jpg', img_copy)
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

def split_by_points(objp, imgp, t_split, shift):
    # Get list of 3D and 2D points and reduce them to a fraction by randomly selecting targets (independent for each frame)
    nobj = []
    nimg = []
    for i in range(len(objp)):
        # Randomize row positions keeping the relation between 3D and 2D points
        c = list(zip(objp[i], imgp[i]))
        random.shuffle(c)
        op, ip = zip(*c)

        # Determine the margins of the new list
        tsp = int(len(objp[i])/t_split)
        sft_start = tsp*shift
        if shift < t_split-1:
            sft_end = tsp*(shift+1)
        else:
            sft_end = len(op)

        # Add the new reduced lists to the major one
        nobj.append(np.array(op[sft_start:sft_end]))
        nimg.append(np.array(ip[sft_start:sft_end]))
    return nobj, nimg

def split_by_distance(objpts, imgpts, names, vecs, min_dist, dist_shift):
    # Get distance of the camera between frames using rvec and tvec and return the lists of frames with a difference over "min_dist".
    arg_split = []
    init_shift = False
    for i in range(len(vecs)):
        rvec = vecs[i][0]
        tvec = vecs[i][1]
        rmat = R.from_rotvec(rvec.reshape(3))
        tmat = np.dot(rmat.as_matrix(), tvec)
        if i == 0:
            tmat_old = tmat
            arg_split.append(i)
        else:
            dtc = np.linalg.norm(tmat_old - tmat)
            if dtc >= min_dist or (dtc >= dist_shift and not init_shift):
                init_shift = True
                tmat_old = tmat
                arg_split.append(i)
    
    # After getting all frames with a significant distance, filter the 3D, 2D and name lists to have only them.  
    nobj = [objpts[i] for i in range(len(objpts)) if i in arg_split]
    nimg = [imgpts[i] for i in range(len(imgpts)) if i in arg_split]
    nnames = [names[i] for i in range(len(names)) if i in arg_split]
    return nobj, nimg, nnames

def adjust_gamma(image, gamma=1.0):
	# Adjust gamma value of the image 
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv.LUT(image, table)

###############################################################################################################################
# Get parser arguments
args = parser.parse_args()

## Parameters
# Import 3D points data
if args.data_3d[-4:] == 'csv':
    # Read .csv and save it as a dataframe
    obj_3D = pd.read_csv(args.data_3d)[['X','Y','Z']]
else: # for txt files without header
    # Read .txt and save it as a dataframe
    header = ['X', 'Y', 'Z', 'sigmaX', 'sigmaY', 'sigmaZ', 'offset']
    obj_3D = pd.read_csv(args.data_3d, delimiter='\t', names=header, index_col=0)[['X','Y','Z']]

# Import 2D point data
with open(args.data_2d) as json_file:
    frame_dict = json.load(json_file)

# Use CODETARGET values from ct_startframe (default=image0)
ct_startframe = frame_dict["CODETARGETS"]

# Get list of anchor points (used to project the rest of the points in a frame)
link_targets = frame_dict["LINKPOINTS"]

# Get list of target exceptions from the 3D point data to project over the 2D data
exception_targets = frame_dict["EXCEPTIONS"]


# Remove all exception targets from the 3D data list
str_exceptions = '|'.join(exception_targets)
flt = obj_3D.index.str.contains(str_exceptions)
obj_3D = obj_3D[~flt]

# Shift all 3D points' positions around the point of interest
points_3D = obj_3D.to_numpy() # BEWARE: to_numpy() doesn't generate a copy but another instance to access the same data. So, if points_3D changes, obj3D will too.
POI = obj_3D.loc[[frame_dict["POI"]]].to_numpy()[0]
points_3D -= POI

## Crossmatch
# Initialize crossmatching algorithm functions
orb = cv.ORB_create(WTA_K=4, nfeatures=10000, edgeThreshold=31, patchSize=255)
bf = cv.BFMatcher.create(cv.NORM_HAMMING2, crossCheck=True)

if args.calibfile:
    cam = camera.Camera(args.calibfile)
    camera_matrix = cam.cam_matrix()
    dist_coeff = cam.dist_coeff()
else:
    # Camera matrix
    fx = 2569.605957
    fy = 2568.584961
    cx = 1881.56543
    cy = 1087.135376

    camera_matrix = np.array([[fx, 0., cx],
                                [0., fy, cy],
                                [0., 0., 1.]], dtype = "double")

    # Distortion coefficients
    k1 = 0.019473
    k2 = -0.041976
    p1 = -0.000273
    p2 = -0.001083
    k3 = 0.030603

    dist_coeff = np.array(([k1], [k2], [p1], [p2], [k3]))

## Flags
# CALIB_USE_INTRINSIC_GUESS: Calibration needs a preliminar camera_matrix to start (necessary in non-planar cases)
flags_model = cv.CALIB_USE_INTRINSIC_GUESS

###############################################################################################################################
# Main
    
if args.halfway:
    # Load .txt file with some specific frame (halfway in the process) codetarget locations
    hf_dir = os.path.dirname(args.data_2d)+'/backup/'+os.path.basename(args.data_2d)[:-4]+'_'+args.halfway
    with open(hf_dir+'.txt') as json_file:
        frame_dict = json.load(json_file)
    
    # Save starting point
    start_frame = int(frame_dict['last_passed_frame'][1])
    print('Starting frame: ',frame_dict['last_passed_frame'][0])
    frame_dict.pop('last_passed_frame')
    
    # Get point values
    df_frame = pd.DataFrame.from_dict(frame_dict)
    cframes = df_frame.to_numpy().reshape(-1,1,2)

    # Get CODETARGET locations in image
    ct_idx = [df_frame.index.get_loc(idx) for idx in df_frame.index if idx in link_targets]
    ct_names = [idx for idx in df_frame.index if idx in link_targets]
    ct_corners = cframes[ct_idx]
    
    # Get CODETARGET locations in 3D
    ct_points_3D = obj_3D.loc[ct_names].to_numpy()
    
    # Load pkl file with 3D-2D already classified data    
    pFile = pickle.load(open(hf_dir+'.pkl', "rb"))

    # Unpack lists from the .pkl file(s)
    objpoints = pFile['3D_points']
    imgpoints = pFile['2D_points']
    ret_names = pFile['name_points']
    tgt_names = pFile['name_targets']
    vecs = pFile['rt_vectors']

else:
    # Set start frame
    start_frame = 0
    
    # Get CODETARGET locations in image
    ct_corners = np.array(list(ct_startframe.values()), dtype=np.float64)
    
    # Get CODETARGET locations in 3D
    ct_points_3D = obj_3D.loc[ct_startframe.keys()].to_numpy()
    
    # Arrays to store object points and image points from all frames possible.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    ret_names = [] # names of every frame for tabulation
    tgt_names = [] # names of every target for correlation
    vecs = []      # rotation and translation vectors from reconstruction


###############################################################################################################################
# Initial process

# Get images from directory
print(f"Searching images in ./sets/{args.folder}/")
images = sorted(glob.glob(f'./sets/{args.folder}/*.jpg'), key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

###############################################################################################################################
# Image processing

pbar = tqdm(desc='READING FRAMES', total=len(images), unit=' frames', dynamic_ncols=True)
if start_frame != 0:
    pbar.update(start_frame)
for fname in images[start_frame:]:
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
    thr = cv.adaptiveThreshold(img_l, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 51, -64)
    
    ## Frame to frame reprojection
    if fname == images[start_frame]:
        #### TO DO ####
        # Make standarized way to find at least the code targets (maybe based on a patch or something)
        # Dilate image to merge all 8 points into one blob for each codetarget to find its center, based on the size of the points (standard distance between points in codetargets)
        # Find a way to compensate when there are no enough points within the window to continue projecting the points in the next frame.
        ###############
        try:              
            wd = 49 # Window size to find and correct codetarget points
            rads = 0
            for ct in ct_corners:
                x_min = 0 if int(ct[0] - wd) <= 0 else int(ct[0] - wd)
                x_max = w if int(ct[0] + wd) >= w else int(ct[0] + wd)
                y_min = 0 if int(ct[1] - wd) <= 0 else int(ct[1] - wd)
                y_max = h if int(ct[1] + wd) >= h else int(ct[1] + wd)
                
                ct_patch = thr[y_min:y_max, x_min:x_max]
                contours, _ = cv.findContours(ct_patch,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
                
                # Calculate centroids of the contours
                centroids = []
                rad = wd
                for contour in contours:
                    M = cv.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        centroids.append([cX, cY])
                centroids = np.array(centroids[1:])
                
                # Get average pixel distance between the points of CODETARGETS
                distances = distance.cdist(centroids, centroids, 'euclidean')
                np.fill_diagonal(distances, np.inf)
                min_distances = np.min(distances, axis=1)
                rads = np.max(min_distances.mean())
        
            # Use the average pixel distance to create the kernel to merge codetarget points                        
            ksize = int(np.mean(rads))
        except:
            ksize = 15
            
        # Create the kernel and apply it over the inverted threshold (for cv.dilate to work as intended)
        kernel = np.ones((ksize, ksize)) 
        thr_new = cv.dilate(cv.bitwise_not(thr), kernel, iterations=1)
        thr_dil = cv.bitwise_not(thr_new)     
        
        ########
        # plt.figure()
        # plt.imshow(thr)
        
        # # if images.index(fname) >= 28:
        # fig, ax = plt.subplots()
        # ax.imshow(thr_dil)
        # ax.scatter(ct_corners[:,0], ct_corners[:,1])
        # for kt in range(ct_corners.shape[0]):
        #     circle = patches.Circle((ct_corners[kt,0], ct_corners[kt,1]), radius=24, edgecolor='r', facecolor='none')
        #     ax.add_patch(circle)
        # plt.show()
        ########
        
    else:
        # Detect new position of CODETARGETS
        kp1, des1 = orb.detectAndCompute(img_old,None) 
        kp2, des2 = orb.detectAndCompute(img_gray,None)
        
        # Match descriptors.
        matches = bf.match(des1,des2)
        dmatches = sorted(matches, key=lambda x:x.distance)
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)
        
        # img3 = cv.drawMatches(img_old,kp1,img_gray,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # displayImage(img3)
        
        # Find homography matrix and do perspective transform to ct_points
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        ct_corners_proy = cv.perspectiveTransform(ct_corners, M)
        
        # Remove CODETARGETS if reprojections are not inside the image
        nn_ct_proy = [ct_corners_proy[i] for i in range(ct_corners_proy.shape[0]) if ct_corners_proy[i,0,0] > 0 and ct_corners_proy[i,0,1] > 0]
        nn_ct_name = [ct_corners_names[i] for i in range(ct_corners_proy.shape[0]) if ct_corners_proy[i,0,0] > 0 and ct_corners_proy[i,0,1] > 0]
                
        ct_corners_proy = np.array(nn_ct_proy)
        ct_corners_names = nn_ct_name
        
        # Find the correct position of points using a small window and getting the highest value closer to the center.
        wd = 49 #24
        ct_corners = []
        ct_names_fix = []
        for i in range(ct_corners_proy.shape[0]):
            cnr = ct_corners_proy[i]
            x_min = 0 if int(cnr[0,0] - wd) <= 0 else int(cnr[0,0] - wd)
            x_max = w if int(cnr[0,0] + wd) >= w else int(cnr[0,0] + wd)
            y_min = 0 if int(cnr[0,1] - wd) <= 0 else int(cnr[0,1] - wd)
            y_max = h if int(cnr[0,1] + wd) >= h else int(cnr[0,1] + wd)
            
            contours, _ = cv.findContours(thr[y_min:y_max, x_min:x_max],cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 1: # and 'CODE' in ct_corners_names[i]) or (len(contours) > 1 and 'TARGET' in ct_corners_names[i]):
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
    ## Find rest of points using CODETARGET projections
    # Get angle of camera by matching known 2D points with 3D points
    res, rvec, tvec = cv.solvePnP(ct_points_3D, ct_corners, camera_matrix, dist_coeff)
    
    # Make simulated image with 3D points data
    points_2D = cv.projectPoints(points_3D, rvec, tvec, camera_matrix, dist_coeff)[0]
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

    # Get position of CODETARGETS plus other important coordinates
    ct_corners_idx = [df_corners.index.get_loc(idx) for idx in df_corners.index if idx in link_targets]
    ct_corners_names = [idx for idx in df_corners.index if idx in link_targets]
    ct_corners = new_corners[ct_corners_idx]
    
    # Show or save frames with points
    if args.plot:
        displayImageWPoints(img0, df_corners, name=ffname, show_names=True, save=True, fdir=args.folder)

    # Save 3D and 2D point data for calibration
    if not (args.halfway and start_frame == images.index(fname)):
        objpoints.append(new_obj3D)
        imgpoints.append(new_corners)
        ret_names.append(ffname)
        tgt_names.append(df_corners.index.to_list())
        vecs.append(np.array([rvec, tvec]))
    
    # Backup
    backup_dir = f'{os.path.dirname(args.data_2d)}/backup'
    if not os.path.exists(backup_dir):
        os.mkdir(backup_dir)
    
    if images.index(fname) % 100 == 0 and start_frame != images.index(fname):
        save_corners = df_corners.to_dict()
        save_corners['last_passed_frame'] = [os.path.basename(fname), images.index(fname)]
        with open(f'{backup_dir}/{os.path.basename(args.data_2d)[:-4]}_f{images.index(fname)}b.txt', 'w') as fp:
            json.dump(save_corners, fp, indent=4)

        vid_data = {'3D_points': objpoints, '2D_points': imgpoints, 'name_points': ret_names, 'name_targets': tgt_names, 'rt_vectors': vecs}
        with open(f'{backup_dir}/{os.path.basename(args.data_2d)[:-4]}_f{images.index(fname)}b.pkl', 'wb') as fpkl:
            pickle.dump(vid_data, fpkl)

    img_old = img_gray
    thr_old = thr
    pbar.update(1)
pbar.close()

if args.save:
    # Save target data in pkl to analyze it in other files.
    # Note: Data will not complete if argparse option '--halfway' is used.
    vid_data = {'3D_points': objpoints, '2D_points': imgpoints, 'name_points': ret_names, 'name_targets': tgt_names, 'rt_vectors': vecs,
                'init_mtx': camera_matrix, 'init_dist': dist_coeff, 'img_shape': img0.shape[1::-1], 'init_calibfile': args.calibfile}
    with open(f'./sets/{args.folder}_vidpoints.pkl', 'wb') as fp:
        pickle.dump(vid_data, fp)
        print(f"Dictionary successfully saved as '{args.folder}_vidpoints.pkl'")

# When everything done, release the frames
cv.destroyAllWindows()

###########################################################################################################################
# Filter data and calibration

if args.calibenable:
    # Define what filters are active or not and put them in the summary.
    summary = ''
    date_today = str(datetime.datetime.now())[5:].split('.')[0].replace('-', '').replace(':', '').replace(' ', '_')
    
    # Filter lists if required
    if args.filterpnts:
        print(f'Filter by points enabled')
        objpoints, imgpoints = split_by_points(objpoints, imgpoints, t_split=args.split, shift=args.shift)
        summary += f'Filter by points, sp={args.split}, sf={args.shift}. '

    if args.filterdist:
        print(f'Filter by distance enabled')
        objpoints, imgpoints, ret_names = split_by_distance(objpoints, imgpoints, ret_names, vecs, min_dist=args.mindist, dist_shift=args.distshift)
        summary += f'Filter by distance, md={args.mindist}, ds={args.distshift}. '
        
    if args.filtertime:
        print(f'Filter by time enabled')
        objpoints = objpoints[args.residue::args.reduction]
        imgpoints = imgpoints[args.residue::args.reduction]
        ret_names = ret_names[args.residue::args.reduction]
        summary += f'Filter by time, rd={args.reduction}, rs={args.residue}. '
        
    summary = summary[:-1]
    print(f'Length of lists for calibration: {len(ret_names)}')
    
    # Camera Calibration
    print("Calculating camera parameters...")
    ret, mtx, dist, rvecs, tvecs, stdInt, stdExt, pVE = cv.calibrateCameraExtended(objpoints, imgpoints, img0.shape[1::-1], cameraMatrix=camera_matrix, distCoeffs=dist_coeff, flags=flags_model)
    pVE_extended = np.array((np.array(ret_names, dtype=object), pVE[:,0])).T
    pVE_extended = pVE_extended[pVE_extended[:,1].argsort()]

    print('Camera matrix:\n', mtx)
    print('Distortion coefficients:\n', dist)
    print('Error per frame:\n', pVE_extended)

    if args.calibsave:
        print(summary)
        if not os.path.exists('./results'):
            os.mkdir('./results')
        fs = cv.FileStorage('./results/'+args.folder+'-'+date_today+'.yml', cv.FILE_STORAGE_WRITE)
        fs.write('summary', summary)
        fs.write('init_cam_calib', os.path.basename(args.calibfile))
        fs.write('camera_matrix', mtx)
        fs.write('dist_coeff', dist)
        fs.write('rvec', np.array(rvecs))
        fs.write('tvec', np.array(tvecs))
        fs.write('std_intrinsics', stdInt)
        fs.write('std_extrinsics', stdExt)
        pVElist = np.array((np.array([int(x[5:]) for x in ret_names]), pVE[:,0])).T
        fs.write('per_view_errors', pVElist)
        fs.release()

print("We finished!")