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
import datetime as dt
import camera
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.spatial import distance



def displayImage(img, width=1280, height=720, name='Picture'):
    # Small simple function to display images without needing to add the auxiliar functions. By default it reduces the size of the image to 1280x720.
    cv.imshow(name, cv.resize(img, (width, height)))
    cv.waitKey(0)
    cv.destroyAllWindows()
        
def displayImageWPoints(img, *args, name='Image', show_names=False, save=False, fdir='new_set'):
    random.seed(0)
    
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
        ffolder, fvid = fdir.split('\\')
        if not os.path.exists(f"{ffolder}/{fvid}/tracked/"):
            os.mkdir(f"{ffolder}/{fvid}/tracked/")
        cv.imwrite(f'{ffolder}/{fvid}/tracked/{name}.jpg', img_copy)
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

def adjust_gamma(image, gamma=1.0):
	# Adjust gamma value of the image 
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv.LUT(image, table)

def points_tracker(folder, data_3d, data_2d, calibfile=None, halfway=None, save_tracked=False, plot=False, gamma=False):
    """
    This script performs camera calibration using images of a GSI-based board. It can detect targets, save data, and execute the calibration process. 
    Saved data (pkl file) can later be used with "calibration.py".

    Parameters:
    - folder (str): Name of the folder containing the frame images (*.jpg).
    - data_3d (str): File containing the 3D positions of TARGETS and CODETARGETS (*.csv, Cartesian units).
    - data_2d (str): File containing the 2D positions of CODETARGETS from the first frame (*.txt, pixel units).
    - calibfile (str): Calibration results file (*.txt), used for point reprojection.
    - halfway (str): Specifies the last part of a file name to restart tracking from any frame.
    - save_tracked (flag): Saves all frames with tracked points in a subfolder.
    - plot (flag): Displays plots of image points and projected points for every frame.
    - gamma (float): Apply gamma correction to the frame.
    """
    
    ## Parameters
    # Import 3D points data
    if data_3d[-4:] == 'csv':
        # Read .csv and save it as a dataframe
        obj_3D = pd.read_csv(data_3d)[['X','Y','Z']]
    else: # for txt files without header
        # Read .txt and save it as a dataframe
        header = ['X', 'Y', 'Z', 'sigmaX', 'sigmaY', 'sigmaZ', 'offset']
        obj_3D = pd.read_csv(data_3d, delimiter='\t', names=header, index_col=0)[['X','Y','Z']]

    # Import 2D point data
    with open(data_2d) as json_file:
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

    if calibfile:
        cam = camera.Camera(calibfile)
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
        
    if halfway:
        # Load .txt file with some specific frame (halfway in the process) codetarget locations
        hf_dir = os.path.dirname(data_2d)+'/backup/'+os.path.basename(data_2d)[:-4]+'_'+halfway
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
    frames_path = os.path.normpath(folder)
    print(f"Searching images in {frames_path}")

    images = sorted(glob.glob(f'{frames_path}/*.jpg'), key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

    ###############################################################################################################################
    # Image processing

    pbar = tqdm(desc='READING FRAMES', total=len(images), unit=' frames', dynamic_ncols=True, miniters=1)
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
        if gamma:
            img_l = adjust_gamma(img_lab[:,:,0], gamma=gamma)
            img_gray = adjust_gamma(img_lab[:,:,0], gamma=gamma)
        else:
            img_l = img_lab[:,:,0]
        
        # Applying threshold to find points
        thr = cv.adaptiveThreshold(img_l, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 51, -64)
        
        ## Frame to frame reprojection
        if fname == images[start_frame]:
            dummy = 'You dummy. Fix this thing!!! Ò_Ó'
            
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
        if plot:
            # Plot only
            displayImageWPoints(img0, df_corners, name=ffname, show_names=True, save=False, fdir=frames_path)
        elif save_tracked:
            # Save only
            displayImageWPoints(img0, df_corners, name=ffname, show_names=True, save=True, fdir=frames_path)

        # Save 3D and 2D point data for calibration
        if not (halfway and start_frame == images.index(fname)):
            objpoints.append(new_obj3D)
            imgpoints.append(new_corners)
            ret_names.append(ffname)
            tgt_names.append(df_corners.index.to_list())
            vecs.append(np.array([rvec, tvec]))
        
        # Backup
        backup_dir = f'{os.path.dirname(data_2d)}/backup'
        if not os.path.exists(backup_dir):
            os.mkdir(backup_dir)
        
        if images.index(fname) % 100 == 0 and start_frame != images.index(fname):
            save_corners = df_corners.to_dict()
            save_corners['last_passed_frame'] = [os.path.basename(fname), images.index(fname)]
            with open(f'{backup_dir}/{os.path.basename(data_2d)[:-4]}_f{images.index(fname)}.txt', 'w') as fp:
                json.dump(save_corners, fp, indent=4)

            vid_data = {'3D_points': objpoints, '2D_points': imgpoints, 'name_points': ret_names, 'name_targets': tgt_names, 'rt_vectors': vecs}
            with open(f'{backup_dir}/{os.path.basename(data_2d)[:-4]}_f{images.index(fname)}.pkl', 'wb') as fpkl:
                pickle.dump(vid_data, fpkl)

        img_old = img_gray
        thr_old = thr
        pbar.update(1)
    pbar.close()

    # Save target data in pkl to analyze it in other files.
    # Note: Data will not complete if argparse option '--halfway' is used.
    vid_data = {'3D_points': objpoints, '2D_points': imgpoints, 'name_points': ret_names, 'name_targets': tgt_names, 'rt_vectors': vecs,
                'init_mtx': camera_matrix, 'init_dist': dist_coeff, 'img_shape': img0.shape[1::-1], 'init_calibfile': calibfile}
    with open(f'{frames_path}_vidpoints.pkl', 'wb') as fp:
        pickle.dump(vid_data, fp)
        print(f"Dictionary successfully saved as '{frames_path}_vidpoints.pkl'")

    # When everything done, release the frames
    cv.destroyAllWindows()

    print("We finished!")
    
    
if __name__ == '__main__':
    # Initialize parser
    parser = argparse.ArgumentParser(description='Camera calibration using GSI-based board images. It allows to find targets, save data and/or do the calibration process. Saved data can be used with "calibration.py".')
    parser.add_argument('folder', type=str, help='Name of the folder containing the frames (*.jpg).')
    parser.add_argument('data_3d', type=str, help='Name of the file containing the 3D position of TARGETS and CODETARGETS (*.csv, cartesian units).')
    parser.add_argument('data_2d', type=str, help='Name of the file containing the 2D position of CODETARGETS of the first frame to analyze. (*.txt, (x,y) pixel units).')
    # Point tracking settings
    parser.add_argument('-gm', '--gamma', type=float, default=False, help='Gamma correction')
    parser.add_argument('-cb', '--calibfile', type=str, metavar='file', help='Name of the file containing calibration results (*.txt), for point reprojection and/or initial guess during calibration.')
    parser.add_argument('-hf', '--halfway', type=str, metavar='target_data', help='Last part of the name of the file containing target data to restart tracking process from any frame.')
    parser.add_argument('-st', '--save_tracked', action='store_true', default=False, help='Saves all frames with their respective tracked points in a created subfolder.')
    parser.add_argument( '-p', '--plot', action='store_true', default=False, help='Shows plots of every frame with image points and projected points.')
    
    # Get parser arguments
    args = parser.parse_args()
    points_tracker(args.folder, args.data_3d, args.data_2d, args.calibfile, args.halfway, args.save_tracked, args.plot, args.gamma)