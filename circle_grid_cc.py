#!/usr/bin/env python

################################################################################
#      Copyright [2017] [ShenZhen Longer Vision Technology], Licensed under    #
#      ******** GNU General Public License, version 3.0 (GPL-3.0) ********     #
#      You are allowed to use this file, modify it, redistribute it, etc.      #
#      You are NOT allowed to use this file WITHOUT keeping the License.       #
#                                                                              #
# Author:           JIA Pei                                                    #
# Contact:          jiapei@longervision.com                                    #
# Create Date:      2017-03-20                                                 #
################################################################################

# Standard imports
import numpy as np
import cv2 as cv
import glob
import argparse
from tqdm import tqdm

# Initialize parser
parser = argparse.ArgumentParser(description='Camera calibration using chessboard images.')
parser.add_argument('folder', type=str, help='Name of the folder containing the frames (*.jpg).')
parser.add_argument('-s', '--scale', type=int, metavar='N', default=0, choices=range(100), help='Scales down the image to get faster (and less reliable) results (range=0:9, default=0) .')
parser.add_argument('-e', '--extended', action='store_true', default=False, help='Enables use of cv.calibrateCameraExtended instead of the default function.')

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

########################################Blob Detector##############################################
# Setup SimpleBlobDetector parameters.
blobParams = cv.SimpleBlobDetector_Params()

# Change thresholds
blobParams.minThreshold = 8
blobParams.maxThreshold = 255

# Filter by Area.
blobParams.filterByArea = True
blobParams.minArea = 64     # minArea may be adjusted to suit for your experiment
blobParams.maxArea = 2500   # maxArea may be adjusted to suit for your experiment

# Filter by Circularity
blobParams.filterByCircularity = True
blobParams.minCircularity = 0.1

# Filter by Convexity
blobParams.filterByConvexity = True
blobParams.minConvexity = 0.87

# Filter by Inertia
blobParams.filterByInertia = True
blobParams.minInertiaRatio = 0.01

# Create a detector with the parameters
blobDetector = cv.SimpleBlobDetector_create(blobParams)
###################################################################################################
# Original blob coordinates, supposing all blobs are of z-coordinates 0
# And, the distance between every two neighbour blob circle centers is 72 centimetres
# In fact, any number can be used to replace 72.
# Namely, the real size of the circle is pointless while calculating camera calibration parameters.
objp = np.zeros((44, 3), np.float32)
objp[0]  = (0, 0, 0)
objp[1]  = (0, 10, 0)
objp[2]  = (0, 20, 0)
objp[3]  = (0, 30, 0)
objp[4]  = (5, 5, 0)
objp[5]  = (5, 15, 0)
objp[6]  = (5, 25, 0)
objp[7]  = (5, 35, 0)
objp[8]  = (10, 0, 0)
objp[9]  = (10, 10, 0)
objp[10] = (10, 20, 0)
objp[11] = (10, 30, 0)
objp[12] = (15, 5,  0)
objp[13] = (15, 15, 0)
objp[14] = (15, 25, 0)
objp[15] = (15, 35, 0)
objp[16] = (20, 0, 0)
objp[17] = (20, 10, 0)
objp[18] = (20, 20, 0)
objp[19] = (20, 30, 0)
objp[20] = (25, 5, 0)
objp[21] = (25, 15, 0)
objp[22] = (25, 25, 0)
objp[23] = (25, 35, 0)
objp[24] = (30, 0, 0)
objp[25] = (30, 10, 0)
objp[26] = (30, 20, 0)
objp[27] = (30, 30, 0)
objp[28] = (35, 5, 0)
objp[29] = (35, 15, 0)
objp[30] = (35, 25, 0)
objp[31] = (35, 35, 0)
objp[32] = (40, 0, 0)
objp[33] = (40, 10, 0)
objp[34] = (40, 20, 0)
objp[35] = (40, 30, 0)
objp[36] = (45, 5, 0)
objp[37] = (45, 15, 0)
objp[38] = (45, 25, 0)
objp[39] = (45, 35, 0)
objp[40] = (50, 0, 0)
objp[41] = (50, 10, 0)
objp[42] = (50, 20, 0)
objp[43] = (50, 30, 0)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.
###################################################################################################

def main():
    args = parser.parse_args()

    images = glob.glob('./sets/'+args.folder+'/*.jpg')
    print(f"Searching images in ./sets/{args.folder}/")

    if args.scale > 0:
        print(f'Scale reduction set to {args.scale}%') 
    if args.extended:
        print(f'CameraCalibrationExtended function set')
        
    ret_detect = [0,0]
    ret_names = []

    pbar = tqdm(desc='READING FRAMES', total=len(images), unit=' frames', dynamic_ncols=True)
    for fname in images:
        img = cv.imread(fname)
        
        # resize image
        if args.scale != 0:
            width = int(img.shape[1] * (1-args.scale/100))
            height = int(img.shape[0] * (1-args.scale/100))
            dim = (width, height)
            img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
            
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        keypoints = blobDetector.detect(gray) # Detect blobs.

        # Draw detected blobs as red circles. This helps cv.findCirclesGrid() . 
        im_with_keypoints = cv.drawKeypoints(img, keypoints, np.array([]), (0,255,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        im_with_keypoints_gray = cv.cvtColor(im_with_keypoints, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findCirclesGrid(im_with_keypoints, (4,11), None, flags = cv.CALIB_CB_ASYMMETRIC_GRID)   # Find the circle grid

        if ret == True:
            objpoints.append(objp)  # Certainly, every loop objp is the same, in 3D.

            corners2 = cv.cornerSubPix(im_with_keypoints_gray, corners, (11,11), (-1,-1), criteria)    # Refines the corner locations.
            imgpoints.append(corners2)

            # Draw and display the corners.
            im_with_keypoints = cv.drawChessboardCorners(img, (4,11), corners2, ret)
            
            # Save some data
            ret_names.append(fname[8+len(args.folder):-4])
            ret_detect[0] += 1
        else:
            ret_detect[1] += 1
        pbar.update(1)
    pbar.close()

    # When everything done, release the capture
    cv.destroyAllWindows()

    print("Calculating camera matrix...")

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
