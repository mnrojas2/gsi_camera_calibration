#!/usr/bin/env python

import os
import argparse
import cv2 as cv
import numpy as np
import pickle
import random
import glob
import re
import datetime
import camera
from scipy.spatial.transform import Rotation as R


# Initialize parser
parser = argparse.ArgumentParser(description='Camera calibration using 2D and 3D data saved in .pkl files.')
parser.add_argument('file', type=str, help='Name of the file containing data (*.pkl).')
parser.add_argument( '-a', '--all', action='store_true', default=False, help='Enables search of all (*.pkl) files to use them for the calculation of a single calibration.')
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

###############################################################################################################################
# Functions

random.seed(0)

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

###############################################################################################################################
# Flags

# CALIB_USE_INTRINSIC_GUESS: Calibration needs a preliminar camera_matrix to start (necessary in non-planar cases)
flags_model = cv.CALIB_USE_INTRINSIC_GUESS

###############################################################################################################################
# Main

# Get parser arguments
args = parser.parse_args()

if args.file.endswith('.pkl'):
    args.all = False

# Load pickle file
if args.all:
    print(f'Loading all .pkl files in {args.file}')
    pkl_list = sorted(glob.glob(f'{args.file}/*.pkl'), key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
else:
    print(f'Loading {args.file}')
    pkl_list = [args.file]

# Define what filters are active or not and put them in the summary.
summary = ''
date_today = str(datetime.datetime.now())[5:].split('.')[0].replace('-', '').replace(':', '').replace(' ', '_')

if args.filterpnts:
    print(f'Filter by points enabled')
    summary += f'Filter by points, sp={args.split}, sf={args.shift}. '

if args.filterdist:
    print(f'Filter by distance enabled')
    summary += f'Filter by distance, md={args.mindist}, ds={args.distshift}. '
    
if args.filtertime:
    print(f'Filter by time enabled')
    summary += f'Filter by time, rd={args.reduction}, rs={args.residue}. '
    
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
    item_name = item.split('\\')[-1][:5]

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
    if args.filterpnts:
        # Filter lists by points
        objpoints, imgpoints = split_by_points(objpoints, imgpoints, t_split=args.split, shift=args.shift)

    if args.filterdist:
        # Filter lists by distance
        objpoints, imgpoints, ret_names = split_by_distance(objpoints, imgpoints, ret_names, vecs, min_dist=args.mindist, dist_shift=args.distshift)
        
    if args.filtertime:
        # Filter lists by time (number of frames)
        objpoints = objpoints[args.residue::args.reduction]
        imgpoints = imgpoints[args.residue::args.reduction]
        ret_names = ret_names[args.residue::args.reduction]
    
    sft = 0
    obj_list += objpoints[sft::len(pkl_list)]
    img_list += imgpoints[sft::len(pkl_list)]
    ret_list += ret_names[sft::len(pkl_list)]

objpoints = obj_list
imgpoints = img_list
ret_names = ret_list

if args.all:
    summary += f' Vidsft={sft}.'

if args.calibfile:
    cam = camera.Camera(args.calibfile)
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

# Camera Calibration
print("Calculating camera parameters...")

ret, mtx, dist, rvecs, tvecs, stdInt, stdExt, pVE = cv.calibrateCameraExtended(objpoints, imgpoints, img_shape, cameraMatrix=camera_matrix, distCoeffs=dist_coeff, flags=flags_model)
pVE_extended = np.array((np.array(ret_names, dtype=object), pVE[:,0])).T
pVE_extended = pVE_extended[pVE_extended[:,1].argsort()]

# Print relevant results
print('Camera matrix:\n', mtx)
print('Distortion coefficients:\n', dist)
print('Error per frame:\n', pVE_extended)

if args.save:
    # Make the file to save all calibration data results
    print(summary)
    if not os.path.exists('./results'):
        os.mkdir('./results')
    if args.all:
        fs = cv.FileStorage('./results/Call-'+date_today+'.yml', cv.FILE_STORAGE_WRITE)
    else:
        file = args.file.replace('\\', '/').split('/')[-1][:-14]
        fs = cv.FileStorage('./results/'+file+'-'+date_today+'.yml', cv.FILE_STORAGE_WRITE)
    fs.write('summary', summary)
    fs.write('init_cam_calib', calibfile)
    fs.write('camera_matrix', mtx)
    fs.write('dist_coeff', dist)
    fs.write('rvec', np.array(rvecs))
    fs.write('tvec', np.array(tvecs))
    fs.write('std_intrinsics', stdInt)
    fs.write('std_extrinsics', stdExt)
    pVElist = np.array((np.array([int(x[10:]) if len(pkl_list) > 1 else int(x[5:]) for x in ret_names]), pVE[:,0])).T
    fs.write('per_view_errors', pVElist)
    fs.release()

print("We finished!")