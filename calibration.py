#!/usr/bin/env python

import os
import argparse
import cv2 as cv
import numpy as np
import pickle
import random
import glob
import re
import datetime as dt
import camera
from scipy.spatial.transform import Rotation as R


# Functions
def split_by_points(objp, imgp, split, shift):
    # Get list of 3D and 2D points and reduce them to a fraction by randomly selecting targets (independent for each frame)
    nobj = []
    nimg = []

    # Set seed 0
    random.seed(0)
    
    for i in range(len(objp)):
        # Randomize row positions keeping the relation between 3D and 2D points
        c = list(zip(objp[i], imgp[i]))
        random.shuffle(c)
        op, ip = zip(*c)

        # Determine the margins of the new list
        tsp = int(len(objp[i])/split)
        sft_start = tsp*shift
        if shift < split-1:
            sft_end = tsp*(shift+1)
        else:
            sft_end = len(op)

        # Add the new reduced lists to the major one
        nobj.append(np.array(op[sft_start:sft_end]))
        nimg.append(np.array(ip[sft_start:sft_end]))
    return nobj, nimg


def split_by_distance(objpts, imgpts, names, vecs, min_dist, shift):
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
            if dtc >= min_dist or (dtc >= shift and not init_shift):
                init_shift = True
                tmat_old = tmat
                arg_split.append(i)
    
    # After getting all frames with a significant distance, filter the 3D, 2D and name lists to have only them.  
    nobj = [objpts[i] for i in range(len(objpts)) if i in arg_split]
    nimg = [imgpts[i] for i in range(len(imgpts)) if i in arg_split]
    nnames = [names[i] for i in range(len(names)) if i in arg_split]
    return nobj, nimg, nnames


# Main
def calibrate_dataset(file, calibfile, save=False, filterdist=False, distmin=0, distshift=0, filtertime=False, timesplit=1, timeshift=0, filterpnts=False, pntsplit=1, pntshift=0):
    """
    file: list of file directories or list of a single element containing a folder directory
    calibfile: directory of calibration file
    save: save calibration results as yml
    filterdist: filter frames by distance
    filtertime: filter frames by time
    filterpnts: randomly filter points each frame
    """
    
    # Check if file is a directory or a single file
    if os.path.isdir(file[0]):
        args_all = True
    else:
        args_all = False

    # Load pickle file(s)
    if args_all:
        print(f'Loading all .pkl files in {file[0]}')
        pkl_list = sorted(glob.glob(f'{file[0]}/*.pkl'), key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
    else:
        if len(file) <= 1:
            print(f'Loading {file[0]}')
        else:
            print(f'Loading {', '.join(file)}.')
        pkl_list = file

    # Define what filters are active or not and put them in the summary.
    summary = ''
    date_today = str(dt.datetime.now())[5:].split('.')[0].replace('-', '').replace(':', '').replace(' ', '_')

    if filterpnts:
        print(f'Filter by points enabled')
        summary += f'Filter by points, sp={pntsplit}, sf={pntshift}. '

    if filterdist:
        print(f'Filter by distance enabled')
        summary += f'Filter by distance, md={distmin}, ds={distshift}. '
        
    if filtertime:
        print(f'Filter by time enabled')
        summary += f'Filter by time, rd={timesplit}, rs={timeshift}. '
        
    summary = summary[:-1]

    # Open the .pkl file(s) and filter the images/points. If there are more than one, split some images and merge the result in one list.
    obj_list = []
    img_list = []
    ret_list = []

    camera_matrix = False
    dist_coeff = False
    img_shape = False
    calibfile = False

    for item in pkl_list:
        pFile = pickle.load(open(item, "rb"))
        item_name = os.path.basename(item).split('_')[0]

        # Unpack lists from the .pkl file(s)
        objpoints = pFile['3D_points']
        imgpoints = pFile['2D_points']
        ret_names = pFile['name_points']
        ret_names = [item_name+name if len(pkl_list) > 1 else name for name in ret_names]

        camera_matrix = pFile['init_mtx']
        dist_coeff = pFile['init_dist']
        img_shape = pFile['img_shape']

        calibfile = pFile['init_calibfile']
        vecs = pFile['rt_vectors']

        # Filter lists if required
        if filterpnts:
            # Filter lists by points
            objpoints, imgpoints = split_by_points(objpoints, imgpoints, t_split=pntsplit, shift=pntshift)

        if filterdist:
            # Filter lists by distance
            objpoints, imgpoints, ret_names = split_by_distance(objpoints, imgpoints, ret_names, vecs, min_dist=distmin, dist_shift=distshift)
            
        if filtertime:
            # Filter lists by time (number of frames)
            objpoints = objpoints[timeshift::timesplit]
            imgpoints = imgpoints[timeshift::timesplit]
            ret_names = ret_names[timeshift::timesplit]
        
        sft = 0
        obj_list += objpoints[sft::len(pkl_list)]
        img_list += imgpoints[sft::len(pkl_list)]
        ret_list += ret_names[sft::len(pkl_list)]

    objpoints = obj_list
    imgpoints = img_list
    ret_names = ret_list

    if args_all:
        summary += f' Vidsft={sft}.'

    if calibfile:
        cam = camera.Camera(calibfile)
        camera_matrix = cam.cam_matrix()
        dist_coeff = cam.dist_coeff()
    else:
        # Use values saved from a previous calibration
        # Camera matrix
        fx = 2569.6059570312500
        fy = 2568.5849609375000
        cx = 1881.5654296875000
        cy = 1087.1353759765625
        
        camera_matrix = np.array([[fx, 0., cx],
                                [0., fy, cy],
                                [0., 0., 1.]], dtype = "double")

        # Distortion coefficients
        k1 = 0.019473474472761
        k2 = -0.041976135224104
        p1 = -0.000272782373941
        p2 = -0.001082847476937
        k3 = 0.030603425577283

        dist_coeff = np.array(([k1], [k2], [p1], [p2], [k3]))

    print(f'Length of lists for calibration: {len(ret_names)}')
    
    # CALIB_USE_INTRINSIC_GUESS: Calibration needs a preliminar camera_matrix to start (necessary in non-planar cases)
    flags_model = cv.CALIB_USE_INTRINSIC_GUESS
    
    # Camera Calibration
    print("Calculating camera parameters...")

    ret, mtx, dist, rvecs, tvecs, stdInt, stdExt, pVE = cv.calibrateCameraExtended(objpoints, imgpoints, img_shape, cameraMatrix=camera_matrix, distCoeffs=dist_coeff, flags=flags_model)
    pVE_extended = np.array((np.array(ret_names, dtype=object), pVE[:,0])).T
    pVE_extended = pVE_extended[pVE_extended[:,1].argsort()]

    # Find the results with worse error (top 5%), remove them from the database, and redo the calibration
    top95_threshold = int(pVE_extended.shape[0] * 0.95)
    top95_names = pVE_extended[top95_threshold:,0]

    # Filter the top 5% and redo the calibration
    objpoints_filt = [item for index, item in enumerate(objpoints) if ret_names[index] not in top95_names]
    imgpoints_filt = [item for index, item in enumerate(imgpoints) if ret_names[index] not in top95_names]
    ret_names = [item for item in ret_names if item not in top95_names]
    
    ret, mtx, dist, rvecs, tvecs, stdInt, stdExt, pVE = cv.calibrateCameraExtended(objpoints_filt, imgpoints_filt, img_shape, cameraMatrix=camera_matrix, distCoeffs=dist_coeff, flags=flags_model)
    pVE_extended = np.array((np.array(ret_names, dtype=object), pVE[:,0])).T
    pVE_extended = pVE_extended[pVE_extended[:,1].argsort()]

    # Print relevant results
    print('Camera matrix:\n', mtx)
    print('Distortion coefficients:\n', dist)
    print('Error per frame:\n', pVE_extended)

    if save:
        # Make the file to save all calibration data results
        print(summary)
        if not os.path.exists('./results'):
            os.mkdir('./results')
        if args_all:
            file = os.path.basename(os.path.normpath(file[0]))                                 # Name of the folder of files
            print(file, file[0], os.path.basename(file[0]))
        else:
            if len(file) <= 1: 
                file = os.path.basename(file[0]).split('_')[0]                                 # Name of the single file
            else:
                file_list = [os.path.basename(sfile).split('_')[0] for sfile in file]
                file = '-'.join(file_list)                                                          # Name of the multiple files with a hyphen in the middle 
        fs = cv.FileStorage('./results/'+file+'-'+date_today+'.yml', cv.FILE_STORAGE_WRITE)
        fs.write('summary', summary)
        fs.write('init_cam_calib', os.path.basename(calibfile))
        fs.write('camera_matrix', mtx)
        fs.write('dist_coeff', dist)
        fs.write('rvec', np.array(rvecs))
        fs.write('tvec', np.array(tvecs))
        fs.write('std_intrinsics', stdInt)
        fs.write('std_extrinsics', stdExt)
        pVElist = np.array((([int(x[1:5] + x.split('frame')[-1].zfill(5)) if len(pkl_list) > 1 else int(x.split('frame')[-1]) for x in ret_names]), pVE[:,0])).T
        fs.write('per_view_errors', pVElist)
        fs.release()
        
        print("File saved successfully!")
        
    else:
        print("Calibration process finished!")
        return mtx, dist, pVE_extended
    
    
    
if __name__=='__main__':
    # Initialize parser
    parser = argparse.ArgumentParser(description='Camera calibration using 2D and 3D data saved in .pkl files.')
    parser.add_argument('file', nargs='+', type=str, help='Name of the file, files or folder containing files that contain data (*.pkl).')
    parser.add_argument('-cb', '--calibfile', type=str, metavar='file', help='Name of the file containing calibration results (*.txt), for point reprojection and/or initial guess during calibration.')
    parser.add_argument( '-s', '--save', action='store_true', default=False, help='Saves calibration data results in .yml format.')
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
    	
    # Get parser arguments
    args = parser.parse_args()

    calibrate_dataset(args.file, args.calibfile, args.filterdist, args.mindist, args.distshift, args.filtertime, args.reduction, args.residue, args.filterpnts, args.split, args.shift)