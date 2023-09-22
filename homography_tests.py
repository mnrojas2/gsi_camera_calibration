#!/usr/bin/env python

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

def displayImage(img, width=1280, height=720, name='Picture'):
    # Small simple function to display images without needing to add the auxiliar functions. By default it reduces the size of the image to 1280x720.
    cv2.imshow(name, cv2.resize(img, (width, height)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def displayImageWPoints(img, pts_list, name='Picture'):
    for i in range(pts_list.shape[0]):
        cv2.circle(img, (int(pts_list[i,0]), int(pts_list[i,1])), 25, (128, 0, 128), -1)
    displayImage(img, name=name)

# CODETARGET index (2D points)
Cindex = ['CODE25', 'CODE26', 'CODE29', 'CODE30', 'CODE31', 'CODE32', 'CODE36', 'CODE38', 'CODE42', 'CODE43', 'CODE45', 'CODE46', 'CODE133', 'CODE134']
    
# Import the 3D points from the csv file
obj_3D = pd.read_csv('./videos/Coords/Bundle.csv')[['X','Y','Z']]
obj_3D = obj_3D.loc[Cindex]

# Point of interest
POI = obj_3D.loc[['CODE45']].to_numpy()[0]

points_3D = obj_3D.to_numpy()
points_3D -= POI

# Camera intrinsic parameters

# Camera matrix
fx = 2605.170124
fy = 2596.136808
cx = 1882.683683
cy = 1072.920820

camera_matrix = np.array([[fx, 0., cx],
                        [0., fy, cy],
                        [0., 0., 1.]], dtype = "double")

# Distortion coefficients
k1 = -0.02247760754865920 # -0.011935952
k2 =  0.48088686640699946 #  0.03064728
p1 = -0.00483894784615441 # -0.00067055
p2 =  0.00201310943773827 # -0.00512621
k3 = -0.38064382293946231 # -0.11974069
# k4 = 
# k5 = 
# k6 =

dist_coeff = np.array(([k1], [k2], [p1], [p2], [k3])) # [k4], [k5], [k6]))

# Extracting points from frames
f230 = np.array([
    [1295,  685], [2559,  255], [1578,  698], [1631, 1076], [1903,  257], [2559, 1085], [ 962,  676], 
    [3140, 1073], [ 943, 1078], [3133,  248], [1907, 1071], [1285, 1072], [1068, 1129], [1539, 1133]
], dtype=np.float64)
nf230 = pd.DataFrame(data=f230, index=Cindex, columns=['x', 'y'])

f1011 = np.array([
    [1151,  988], [1877,  598], [1288,  996], [1314, 1281], [1461,  648], [1849, 1332], [1006,  984],
    [2317, 1357], [ 994, 1257], [2340,  536], [1439, 1289], [1145, 1264], [1025, 1298], [1244, 1321]
], dtype=np.float64)
nf1011 = pd.DataFrame(data=f1011, index=Cindex, columns=['x', 'y'])

img_230 = cv2.imread('./tests/frame230.jpg')
img_1011 = cv2.imread('./tests/frame1011.jpg')
displayImageWPoints(img_1011, f1011)

# Try to make a match of both to check if the solution is in itself consistent
res, rvec0, tvec0 = cv2.solvePnP(points_3D, f1011, camera_matrix, dist_coeff)
r0 = R.from_rotvec(rvec0.flatten())
ang0 = r0.as_euler('XYZ', degrees=True)
print(ang0)