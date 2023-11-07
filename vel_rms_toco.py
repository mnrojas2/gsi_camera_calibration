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
        if not os.path.exists('./tests/tracked-sets/'+fdir):
            os.mkdir('./tests/tracked-sets/'+fdir)
        cv.imwrite(f'./tests/tracked-sets/{fdir}/{name}.jpg', img_copy)
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

#'''
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('frame (i)')
ax1.set_ylabel('RMS Error amplitude (Pixels)')
ax1.plot(np.arange(80), rms_error[:8000:100], color=color) # rms_error.shape[0]
ax1.tick_params(axis='y', labelcolor=color)

color = 'tab:blue'
ax2 = ax1.twinx()
ax2.set_xlabel('frame (i)')
ax2.set_ylabel('Angular velocity (pixels/s)')
ax2.plot(np.arange(80), vel_list[:8000:100], color=color) # vel_list.shape[0]
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Angular Velocity and RMS Error vs time (frames)')
plt.show()

# img0 = cv.imread(f'./sets/{args.file}Finf/{ret_names[rp0]}.jpg')
# displayImageWPoints(img0, proy_points_2D, real_points_2D, name=ffname)

# what's left
# determinar que es ese valor RMS (ojala error en puntos) -> Es el error rms promedio asociado a la distancia del punto en pixeles
# correr los videos del dron y determinar periodos de velocidad angular alto


