#!/usr/bin/env python

import os
import argparse
import pandas as pd
import numpy as np
import cv2 as cv
import pickle
import copy
from scipy.spatial.transform import Rotation as R 
from matplotlib import pyplot as plt

# Initialize parser
parser = argparse.ArgumentParser(description='Camera calibration using chessboard images.')
parser.add_argument('file', type=str, help='Name of the file containing data (*.pkl).')

# Functions
def displayImage(img, width=1280, height=720, name='Picture'):
    # Small simple function to display images without needing to add the auxiliar functions. By default it reduces the size of the image to 1280x720.
    cv.imshow(name, cv.resize(img, (width, height)))
    cv.waitKey(0)
    cv.destroyAllWindows()
        
def displayImageWPoints(img, *args, name='Image', show_names=False, save=False, fdir='new_set'):
    # Create output folder if it wasn't created yet
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
        if not os.path.exists('./sets/tracked-sets/'+fdir):
            os.mkdir('./sets/tracked-sets/'+fdir)
        cv.imwrite(f'./sets/tracked-sets/{fdir}/{name}.jpg', img_copy)
    else:
        displayImage(img_copy, name=name)

###############################################################################################################################
# Main

# Get parser arguments
args = parser.parse_args()

# Load pickle file
print(f'Loading {args.file}.pkl')
pFile = pickle.load(open(f"./datasets/pkl-files/{args.file}.pkl","rb"))

# Unpack lists
pts3D = pFile['3D_points']
pts3D = np.array([item for item in pts3D.tolist() if pts3D.tolist().index(item) not in [1, 11]], dtype=np.float64) if pts3D.shape[0] == 12 else pts3D

imgpoints = pFile['2D_points']

camera_matrix = pFile['init_mtx']
dist_coeff = pFile['init_dist']
img_shape = pFile['img_shape']

calibfile = pFile['init_calibfile']
vecs = pFile['rt_vectors']

vel_list = []
for i in range(len(imgpoints)):
    if i == 0:
        tgt_old = imgpoints[i].reshape(-1, 2)
    
    elif i != 0:
        tgt = imgpoints[i].reshape(-1, 2)

        new_tgt = np.linalg.norm(tgt - tgt_old, axis=1)
        vel_ang = 29.97 * new_tgt.mean()
        vel_list.append(vel_ang)
        
        tgt_old = tgt

vel_list = np.array(vel_list)
 
#############################################################################################

#'''
rms_error = []
rms_names = []

for i in range(len(imgpoints)):
    real_points_2D = imgpoints[i]
    proy_points_2D = cv.projectPoints(objectPoints=pts3D, rvec=vecs[i][0], tvec=vecs[i][1], cameraMatrix=camera_matrix, distCoeffs=dist_coeff)[0]
    dist_pts2D = cv.norm(proy_points_2D.reshape(-1,2), real_points_2D.reshape(-1,2), normType=cv.NORM_L2)
    mean_pts2D = np.sqrt(np.dot(dist_pts2D, dist_pts2D)/real_points_2D.shape[0])
    rms_error.append(mean_pts2D)
    
rms_error = np.array(rms_error)

# Plot curves to check correlation
fig, ax1 = plt.subplots(figsize=(12, 7))

color = 'tab:blue'
ax1.set_xlabel('frame (i)')
ax1.set_ylabel('Angular velocity (degrees/s)')
ax1.plot(vel_list/np.sqrt(camera_matrix[0,0]*camera_matrix[1,1]) * 180/np.pi, color=color) # vel_list.shape[0]
ax1.tick_params(axis='y', labelcolor=color)

color = 'tab:red'
ax2 = ax1.twinx()
ax2.set_xlabel('frame (i)')
ax2.set_ylabel('RMS Error amplitude (Pixels)')
ax2.plot(rms_error, color=color) # rms_error.shape[0]
ax2.tick_params(axis='y', labelcolor=color)


plt.title('Angular Velocity and RMS Error vs time (frames)')
fig.tight_layout()

# function y_vel = m * x_error + b
m = 3.51763
b = 0 # 2.16123

# function inv x_error = minv * y_vel + binv
m_inv = 1/m
b_inv = -b/m

rgr = 8000
x_rms = rms_error[:rgr]
y_vel = (vel_list/np.sqrt(camera_matrix[0,0]*camera_matrix[1,1]) * 180/np.pi)[:rgr]

x_rms_fromvel = m_inv * y_vel + b_inv
y_vel_fromrms = m * x_rms + b

plt.figure(figsize=(12, 7))
plt.scatter(x_rms, y_vel, label='measured data')
plt.plot(x_rms_fromvel, y_vel, color='r', label='y_vel data fit')
plt.plot(x_rms, y_vel_fromrms, color='g', label='x_rms data fit')
plt.xlabel('RMS Error amplitude (Pixels)')
plt.ylabel('Angular velocity (degrees/s)')
plt.legend()
plt.title('Angular velocity vs RMS Error')
plt.tight_layout()

plt.figure(figsize=(12, 7))
plt.hist(y_vel, bins=20, label='y_vel')
plt.hist(x_rms*m+b, bins=20, label='x_rms*m+b')
plt.title('Histograms of Angular velocity and RMS Error')
plt.legend()
plt.tight_layout()
plt.show()

# img0 = cv.imread(f'./sets/{args.file}Finf/{ret_names[rp0]}.jpg')
# displayImageWPoints(img0, proy_points_2D, real_points_2D, name=ffname)

# what's left
# determinar que es ese valor RMS (ojala error en puntos) -> Es el error rms promedio asociado a la distancia del punto en pixeles
# correr los videos del dron y determinar periodos de velocidad angular alto


