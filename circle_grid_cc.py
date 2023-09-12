################################################################################
#                                                                              #
#                                                                              #
#           IMPORTANT: READ BEFORE DOWNLOADING, COPYING AND USING.             #
#                                                                              #
#                                                                              #
#      Copyright [2017] [ShenZhen Longer Vision Technology], Licensed under    #
#      ******** GNU General Public License, version 3.0 (GPL-3.0) ********     #
#      You are allowed to use this file, modify it, redistribute it, etc.      #
#      You are NOT allowed to use this file WITHOUT keeping the License.       #
#                                                                              #
#      Longer Vision Technology is a startup located in Chinese Silicon Valley #
#      NanShan, ShenZhen, China, (http://www.longervision.cn), which provides  #
#      the total solution to the area of Machine Vision & Computer Vision.     #
#      The founder Mr. Pei JIA has been advocating Open Source Software (OSS)  #
#      for over 12 years ever since he started his PhD's research in England.  #
#                                                                              #
#      Longer Vision Blog is Longer Vision Technology's blog hosted on github  #
#      (http://longervision.github.io). Besides the published articles, a lot  #
#      more source code can be found at the organization's source code pool:   #
#      (https://github.com/LongerVision/OpenCV_Examples).                      #
#                                                                              #
#      For those who are interested in our blogs and source code, please do    #
#      NOT hesitate to comment on our blogs. Whenever you find any issue,      #
#      please do NOT hesitate to fire an issue on github. We'll try to reply   #
#      promptly.                                                               #
#                                                                              #
#                                                                              #
# Version:          0.0.1                                                      #
# Author:           JIA Pei                                                    #
# Contact:          jiapei@longervision.com                                    #
# URL:              http://www.longervision.cn                                 #
# Create Date:      2017-03-20                                                 #
# Modified Date:    2023-08-14                                                 #
################################################################################

# Standard imports
import sys
import numpy as np
import cv2
#import yaml
import glob


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

########################################Blob Detector##############################################
# Setup SimpleBlobDetector parameters.
blobParams = cv2.SimpleBlobDetector_Params()

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
blobDetector = cv2.SimpleBlobDetector_create(blobParams)
###################################################################################################


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
###################################################################################################


def main():
    args = sys.argv[1:]
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    folder_location = args[0] #input("Name of the folder? : ")
    images = glob.glob('./sets/'+folder_location+'/*.jpg')
    print(f"Searching images in ./sets/{folder_location}/")

    if len(args) >= 2 and args[1] in ('--scale','-s'):
        scale_reduction = int(args[2]) #int(input("% of reduction for each frame?: ")) #0-99
        print(f'Scale reduction set to {scale_reduction}%')
    else: 
        scale_reduction = 0
        print(f'No scale set')
        
    if len(args) >= 4 and args[3] in ('--extended', '-e'):
        extended = args[4]
        print(f'CameraCalibrationExtended function set')
    else:
        extended = False
        print(f'CameraCalibration function set')
        
    ret_detect = [0,0]
    ret_names = []

    for fname in images:
        img = cv2.imread(fname)
        
        # resize image
        if scale_reduction != 0:
            width = int(img.shape[1] * (1-scale_reduction/100))
            height = int(img.shape[0] * (1-scale_reduction/100))
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        keypoints = blobDetector.detect(gray) # Detect blobs.

        # Draw detected blobs as red circles. This helps cv2.findCirclesGrid() . 
        im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        im_with_keypoints_gray = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findCirclesGrid(im_with_keypoints, (4,11), None, flags = cv2.CALIB_CB_ASYMMETRIC_GRID)   # Find the circle grid

        if ret == True:
            objpoints.append(objp)  # Certainly, every loop objp is the same, in 3D.

            corners2 = cv2.cornerSubPix(im_with_keypoints_gray, corners, (11,11), (-1,-1), criteria)    # Refines the corner locations.
            imgpoints.append(corners2)

            # Draw and display the corners.
            im_with_keypoints = cv2.drawChessboardCorners(img, (4,11), corners2, ret)
            
            # Save some data
            ret_names.append(fname[8+len(args[0]):-4])
            ret_detect[0] += 1
        else:
            ret_detect[1] += 1

        cv2.imwrite("testing/"+fname[16:], im_with_keypoints)
        #cv2.imshow("img", im_with_keypoints) # display
        #cv2.waitKey(0)

    # When everything done, release the capture
    cv2.destroyAllWindows()

    print("Calculating camera matrix...")

    camera_matrix = np.eye(3)
    camera_matrix[0, 0] = 2615.249 # 2650 #2615
    camera_matrix[1, 1] = 2610.074 # 2650 #2615
    camera_matrix[0, 2] = 1888.417 # float(img.shape[1]) / 2.0
    camera_matrix[1, 2] = 1093.929 # float(img.shape[0]) / 2.0
    msg = 'Initial Camera Matrix:\n%s' % np.array2string(camera_matrix)

    flags_model = cv2.CALIB_USE_INTRINSIC_GUESS
    # flags_model |= cv2.CALIB_RATIONAL_MODEL # Enable 6 rotation distortion constants instead of 3

    if not extended:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    else:
        ret, mtx, dist, rvecs, tvecs, stdInt, stdExt, pVE = cv2.calibrateCameraExtended(objpoints, imgpoints, gray.shape[::-1], cameraMatrix=camera_matrix, distCoeffs=None,
                                            flags=flags_model)
        pVE_extended = np.array((np.array(ret_names, dtype=object), pVE[:,0])).T

    #  Python code to write the image (OpenCV 3.2)
    fs = cv2.FileStorage('./results/calibration'+folder_location+'_upd.yml', cv2.FILE_STORAGE_WRITE)
    fs.write('camera_matrix', mtx)
    fs.write('dist_coeff', dist)
    if extended:
        fs.write('std_intrinsics', stdInt)
        fs.write('std_extrinsics', stdExt)
        fs.write('per_view_errors', pVE)
    fs.release()
    
    if extended:
        pVE_extended = pVE_extended[pVE_extended[:,1].argsort()]
        print(pVE_extended)
        print("----------------------")
    print(mtx)
    print("----------------------")
    print("Frames w/detections:",ret_detect[0], "/ Frames wo/detections:",ret_detect[1])
    
    # # Using the derived camera parameters to undistort the image

    # img = cv2.imread(images[0])
    # # Refining the camera matrix using parameters obtained by calibration
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # # Method 1 to undistort the image
    # dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # # Method 2 to undistort the image
    # mapx,mapy=cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)

    # dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
    
if __name__ == '__main__':
    main()
