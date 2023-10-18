import argparse
import numpy as np


import cv2 as cv
import pickle

# Initialize parser
parser = argparse.ArgumentParser(description='Camera calibration using chessboard images.')
parser.add_argument('file', type=str, help='Name of the file containing data (*pkl).')
parser.add_argument( '-e', '--extended', action='store_true', default=False, help='Enables use of cv.calibrateCameraExtended instead of the default function.')
parser.add_argument( '-s', '--save', action='store_true', default=False, help='Saves calibration data results in .yml format as well as TARGET position data in .txt format.')
parser.add_argument('-rd', '--reduction', type=int, metavar='N', default=1, help='Reduction of number of frames (total/N).')
parser.add_argument('-rs', '--residue', type=int, metavar='N', default=0, help='Residue or offset for the reduced number of frames.')

# Flags
# CALIB_USE_INTRINSIC_GUESS: Calibration needs a preliminar camera_matrix to start (necessary in non-planar cases)
flags_model = cv.CALIB_USE_INTRINSIC_GUESS

###############################################################################################################################
# Main

# Get parser arguments
args = parser.parse_args()

# Load pickle file
pFile = pickle.load(open(f"./tests/points-data/tracked/{args.file}.pkl","rb"))

# reduction
rd = args.reduction
rs = args.residue
objpoints = pFile['3D_points'][rs::rd]
imgpoints = pFile['2D_points'][rs::rd]
ret_names = pFile['name_points'][rs::rd]
camera_matrix = pFile['init_mtx']
dist_coeff = pFile['init_dist']
img_shape = pFile['img_shape']

# Camera Calibration
print("Calculating camera parameters...")
if args.extended:
    ret, mtx, dist, rvecs, tvecs, stdInt, stdExt, pVE = cv.calibrateCameraExtended(objpoints, imgpoints, img_shape, cameraMatrix=camera_matrix, distCoeffs=dist_coeff, flags=flags_model)
    pVE_extended = np.array((np.array(ret_names, dtype=object), pVE[:,0])).T
    pVE_extended = pVE_extended[pVE_extended[:,1].argsort()]
else:
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img_shape, cameraMatrix=camera_matrix, distCoeffs=dist_coeff, flags=flags_model)

print('Camera matrix:\n', mtx)
print('Distortion coefficients:\n', dist)
if args.extended:
    print('Error per frame:\n', pVE_extended)

if args.save:
    summary = input("Insert comments: ")
    fs = cv.FileStorage('./tests/results/'+args.folder[:-4]+'.yml', cv.FILE_STORAGE_WRITE)
    fs.write('summary', summary)
    fs.write('init_cam_calib', args.calibfile) # ???????????????????
    fs.write('camera_matrix', mtx)
    fs.write('dist_coeff', dist)
    if args.extended:
        pVElist = np.array((np.array([int(x[5:]) for x in ret_names]), pVE[:,0])).T
        fs.write('std_intrinsics', stdInt)
        fs.write('std_extrinsics', stdExt)
        fs.write('per_view_errors', pVElist)
    fs.release()

print("We finished!") # '''