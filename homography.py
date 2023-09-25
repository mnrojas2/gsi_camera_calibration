#!/usr/bin/env python

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance

#############################################################################
# Functions

def displayImage(img, width=1280, height=720, name='Picture'):
    # Small simple function to display images without needing to add the auxiliar functions. By default it reduces the size of the image to 1280x720.
    cv2.imshow(name, cv2.resize(img, (width, height)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def displayImageWPoints(img, *args, name='Picture'):
    for arg in args:
        for i in range(arg.shape[0]):
            cv2.circle(img, (int(arg[i,0]), int(arg[i,1])), 25, (128, 0, 128), -1)
    displayImage(img, name=name)
    
def scatterPlot(*args):
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    for arg in args:
        ax.scatter(arg[:,0], -arg[:,1])
    plt.show()

#############################################################################
# Blob detection parameters

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Initializing parameter setting using cv2.SimpleBlobDetector function
blobParams = cv2.SimpleBlobDetector_Params()

# Filter by Area
blobParams.filterByArea = True
blobParams.maxArea = 500

# Filter by Circularity
blobParams.filterByCircularity = True
blobParams.minCircularity = 0.9

# Creating a blob detector using the defined parameters
blobDetector = cv2.SimpleBlobDetector_create(blobParams)

#############################################################################
# GSI data import

# CODETARGET index (2D points)
Cindex = ['CODE25', 'CODE26', 'CODE29', 'CODE30', 'CODE31', 'CODE32', 'CODE36', 'CODE38', 'CODE42', 'CODE43', 'CODE45', 'CODE46', 'CODE133', 'CODE134']
    
# Import the 3D points from the csv file
obj_3D = pd.read_csv('./videos/Coords/Bundle.csv')[['X','Y','Z']]
obj_3D_ct = obj_3D.loc[Cindex]
p3D_index = obj_3D.index.to_list()

# Point of interest
POI = obj_3D_ct.loc[['CODE45']].to_numpy()[0]

# Set list of 3D points
points_3D_ct = obj_3D_ct.to_numpy()
points_3D_ct -= POI

points_3D = obj_3D.to_numpy()
points_3D -= POI

#############################################################################
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

#############################################################################
# Main

# Extracting points from frames
ct_f230 = np.array([
    [1295,  685], [2559,  255], [1578,  698], [1631, 1076], [1903,  257], [2559, 1085], [ 962,  676], 
    [3140, 1073], [ 943, 1078], [3133,  248], [1907, 1071], [1285, 1072], [1068, 1129], [1539, 1133]
], dtype=np.float64)
ct_nf230 = pd.DataFrame(data=ct_f230, index=Cindex, columns=['x', 'y'])

ct_f1011 = np.array([
    [1151,  988], [1877,  598], [1288,  996], [1314, 1281], [1461,  648], [1849, 1332], [1006,  984],
    [2317, 1357], [ 994, 1257], [2340,  536], [1439, 1289], [1145, 1264], [1025, 1298], [1244, 1321]
], dtype=np.float64)
ct_nf1011 = pd.DataFrame(data=ct_f1011, index=Cindex, columns=['x', 'y'])

# Read image
# img_230 = cv2.imread('./tests/frame230.jpg')
img_1011 = cv2.imread('./tests/frame1011.jpg')
# displayImageWPoints(img_1011, ct_f1011)

# Get angle of camera by matching known 2D points with 3D points
res, rvec, tvec = cv2.solvePnP(points_3D_ct, ct_f1011, camera_matrix, dist_coeff)
r0 = R.from_rotvec(rvec.flatten())
ang = r0.as_euler('XYZ', degrees=True)
# print(ang)

# Make simulated image with 3D points data
points_2D = cv2.projectPoints(points_3D, rvec, tvec, camera_matrix, dist_coeff)[0]
# scatterPlot(points_2D[:,0,:], ct_f1011, np.array([[1024.61695694,1297.73753909], [1024.61695694,1297.73753909]]))

# for i in range(points_3D_ct.shape[0]):
#     print(Cindex[i], points_3D_ct[i,:], points_2D[i,0,:])
#     # Orden sigue siendo el mismo entre points_3D y points_2D (projectPoints)

# Detect points in image
# call convertScaleAbs function
img_adjusted = cv2.convertScaleAbs(img_1011, alpha=2.0, beta=-50.0) # alpha: contrast, beta: brightness

# Applying threshold to find points
g_thr = 50
_, thr = cv2.threshold(img_adjusted, g_thr, 255, cv2.THRESH_BINARY_INV)
keypoints = blobDetector.detect(thr)

# Creating a list of corners (equivalent of findCirclesGrid)
corners = [[[key.pt[0], key.pt[1]]] for key in keypoints]
corners = np.array(corners, dtype=np.float32)
# displayImageWPoints(img_1011, corners[:,0,:], name='Imagen con puntos')
# scatterPlot(points_2D[:,0,:], corners[:,0,:])

# Get distance between 2D projected points and 2D image points
corners_matrix = distance.cdist(corners[:,0,:], points_2D[:,0,:])

# Convert matrix array in dataframe with proper index and apply idxmin function to find the name of the closest point (p3D_index ~ points_2D)
corners_dataframe = pd.DataFrame(data=corners_matrix, index=np.arange(0, len(corners), 1), columns=p3D_index)
corners_min = corners_dataframe.idxmin(axis='columns')

corners_min.to_csv('datos.txt')

# Eliminar los CODETARGET que están de más (error en GSI ya que no cuenta los puntos alreadedor del central del target)
# Reordenar la lista de los corners, y/o gsi targets para dejar datos 2D y 3D en la misma columna

# Hacer lo mismo para el resto del dataset de imágenes para obtener los sets para calibrar