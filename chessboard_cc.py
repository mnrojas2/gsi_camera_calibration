#!/usr/bin/env python

import cv2 as cv
import numpy as np
import glob
import argparse
from tqdm import tqdm

# Initialize parser
parser = argparse.ArgumentParser(description='Camera calibration using chessboard images.')
parser.add_argument('folder', type=str, help='Name of the folder containing the frames (*.jpg).')
parser.add_argument('-s', '--scale', type=int, metavar='N', default=0, choices=range(100), help='Scales down the image to get faster (and less reliable) results (range=0:9, default=0) .')
parser.add_argument('-e', '--extended', action='store_true', default=False, help='Enables use of cv.calibrateCameraExtended instead of the default function.')

# Defining the dimensions of checkerboard
CHECKERBOARD = (7,10)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

def main():
    args = parser.parse_args()
    
    # Extracting path of individual image stored in a given directory
    images = glob.glob('./sets/'+args.folder+'/*.jpg')
    print(f"Searching images in ./sets/{args.folder}/")

    if args.scale > 0:
        print(f'Scale reduction set to {args.scale}%') 
    if args.extended:
        print(f'CameraCalibrationExtended function set')

    ret_detect = [0,0]
    ret_names = []

    pbar = tqdm(desc='READING FRAMES', total=len(images), unit=' frames')
    for fname in images:
        img = cv.imread(fname)
    
        # resize image
        if args.scale != 0:
            width = int(img.shape[1] * (1-args.scale/100))
            height = int(img.shape[0] * (1-args.scale/100))
            dim = (width, height)
            img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
        
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, cv.CALIB_CB_ADAPTIVE_THRESH+
            cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)
        
        """
        If desired number of corner are detected, we refine the pixel coordinates and display them on the images of checker board
        """
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv.drawChessboardCorners(img, CHECKERBOARD, corners2,ret)
            
            # Save some data
            ret_names.append(fname[8+len(args.folder):-4])
            ret_detect[0] += 1
        else:
            ret_detect[1] += 1
        pbar.update(1)
    pbar.close()

    # When everything done, release the capture
    cv.destroyAllWindows()

    """
    Performing camera calibration by passing the value of known 3D points (objpoints) and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    print("Calculating camera matrix...")
    
    # Initial camera matrix
    camera_matrix = np.eye(3)
    camera_matrix[0, 0] = 2615.249 # 2650 #2615
    camera_matrix[1, 1] = 2610.074 # 2650 #2615
    camera_matrix[0, 2] = 1888.417 # float(img.shape[1]) / 2.0
    camera_matrix[1, 2] = 1093.929 # float(img.shape[0]) / 2.0

    # Flags
    flags_model = cv.CALIB_USE_INTRINSIC_GUESS
    # flags_model |= cv.CALIB_RATIONAL_MODEL # Enable 6 rotation distortion constants instead of 3

    if not args.extended:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    else:
        ret, mtx, dist, rvecs, tvecs, stdInt, stdExt, pVE = cv.calibrateCameraExtended(objpoints, imgpoints, gray.shape[::-1], cameraMatrix=camera_matrix, distCoeffs=None,
                                            flags=flags_model)
        pVE_extended = np.array((np.array(ret_names, dtype=object), pVE[:,0])).T

    #  Python code to write the image (OpenCV 3.2)
    fs = cv.FileStorage('./results/calibration'+args.folder+'.yml', cv.FILE_STORAGE_WRITE)
    fs.write('camera_matrix', mtx)
    fs.write('dist_coeff', dist)
    if args.extended:
        fs.write('std_intrinsics', stdInt)
        fs.write('std_extrinsics', stdExt)
        fs.write('per_view_errors', pVE)
    fs.release()

    if args.extended:
        pVE_extended = pVE_extended[pVE_extended[:,1].argsort()]
        print(pVE_extended)
        print("----------------------")
    print(mtx)
    print("----------------------")
    print(f"Frames w/detections:{ret_detect[0]} / Frames wo/detections:{ret_detect[1]}")
    
    # # Using the derived camera parameters to undistort the image

    # img = cv.imread(images[0])
    # # Refining the camera matrix using parameters obtained by calibration
    # newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # # Method 1 to undistort the image
    # dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    # # Method 2 to undistort the image
    # mapx,mapy=cv.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)

    # dst = cv.remap(img,mapx,mapy,cv.INTER_LINEAR)

if __name__ == '__main__':
    main()