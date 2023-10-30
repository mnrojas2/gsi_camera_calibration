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
parser.add_argument('file', type=str, help='Name of the file containing data (*pkl).')
parser.add_argument('-cb', '--calibfile', type=str, metavar='file', help='Name of the file containing calibration results (*.yml), for point reprojection and/or initial guess during calibration.')


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
        clr = np.array([255, 0, 0])
        if len(args) > 1:
            clr += np.array([-128, 128, 128])
            clr = (np.array(clr) + np.random.randint(-128, 128, size=3)).tolist()
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
print(f'Loading {args.file}Finf_vidpoints.pkl')
pFile = pickle.load(open(f"./datasets/pkl-files/{args.file}Finf_vidpoints.pkl","rb"))

# Unpack lists
objpoints = pFile['3D_points']
imgpoints = pFile['2D_points']
ret_names = pFile['name_points']
tgt_names = pFile['name_targets']

camera_matrix = pFile['init_mtx']
dist_coeff = pFile['init_dist']
img_shape = pFile['img_shape']

calibfile = pFile['init_calibfile']
vecs = pFile['rt_vectors']

vel_list = []
for i in range(len(imgpoints)):
    if i == 0:
        tgt_old = pd.DataFrame(data=imgpoints[i].reshape(-1, 2), index=tgt_names[i], columns=['X', 'Y'])
        # img0 = cv.imread(f'./sets/{args.file[:-10]}/{ret_names[i]}.jpg')
        # displayImageWPoints(img0, tgt_old, name=ret_names[i], show_names=True)
    
    elif i != 0:
        tgt = pd.DataFrame(data=imgpoints[i].reshape(-1, 2), index=tgt_names[i], columns=['X', 'Y'])
        tgt_common = (tgt.index.intersection(tgt_old.index)).tolist()
        # displayImageWPoints(img1, tg1.loc[tgt_common], name=ret_names[i], show_names=True)

        new_tgt = np.linalg.norm(tgt.loc[tgt_common] - tgt_old.loc[tgt_common], axis=1)
        df_tgt = pd.DataFrame(data=new_tgt, index=tgt_common, columns=['dist'])
        vel_ang = (29.97 * df_tgt.mean()).to_numpy()
        vel_list.append(vel_ang)
        
        tgt_old = tgt

df_vel = pd.DataFrame(data=vel_list, index=ret_names[1:], columns=['vel_ang'])

#############################################################################################

print(f'Loading {args.file}-{args.calibfile}.yml')
fs = cv.FileStorage(f'./results/{args.file}-{args.calibfile}.yml', cv.FILE_STORAGE_READ)

mtx = fs.getNode("camera_matrix").mat()
dist_coeff = fs.getNode("dist_coeff").mat()
rvecs = fs.getNode("rvec").mat()
tvecs = fs.getNode("tvec").mat()
pve = fs.getNode("per_view_errors").mat()

pve_keys = ['frame'+str(int(pve[i,0])) for i in range(pve.shape[0])]
df_pve = pd.DataFrame(data=pve[:,1], index=pve_keys, columns=['RMSE'])

vel_pve = (df_vel.index.intersection(df_pve.index)).tolist()

'''
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('frame (i)')
ax1.set_ylabel('RMS Error amplitude (?)')
ax1.plot(pve[1:,0], df_pve.loc[vel_pve].to_numpy(), color=color)
ax1.tick_params(axis='y', labelcolor=color)

color = 'tab:blue'
ax2 = ax1.twinx()
ax2.set_xlabel('frame (i)')
ax2.set_ylabel('Angular velocity (pixels/s)')
ax2.plot(pve[1:,0], df_vel.loc[vel_pve].to_numpy(), color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Angular Velocity vs RMS Error')

plt.figure()
plt.scatter(df_pve.loc[vel_pve].to_numpy(), df_vel.loc[vel_pve].to_numpy())
plt.show() # '''

#'''
rms_error = []
rms_names = []

for i in range(len(imgpoints)):
    ffname = ret_names[i]
    real_points_2D = imgpoints[i]
    for j in range(len(pve[:,0])):
        if ffname == 'frame'+str(int(pve[j,0])):
            proy_points_2D = cv.projectPoints(objectPoints=objpoints[i], rvec=rvecs[j], tvec=tvecs[j], cameraMatrix=camera_matrix, distCoeffs=dist_coeff)
            
            
            
            dist_pts2D = np.linalg.norm(proy_points_2D.reshape(-1,2) - real_points_2D.reshape(-1,2), axis=1)
            mean_pts2D = np.mean(dist_pts2D)
            rms_error.append(mean_pts2D)
            print(ffname, pve[j,1], mean_pts2D, mean_pts2D/pve[j,1])
            rms_names.append(ffname)
    
df_rms = pd.DataFrame(data=np.array(rms_error), index=rms_names, columns=['Error'])

rms_pve = (df_rms.index.intersection(df_pve.index)).tolist() # '''

# plt.figure()
# plt.plot(pve[:,0], df_rms.loc[rms_pve].to_numpy())
# plt.plot(pve[:,0], df_pve.loc[rms_pve].to_numpy())
# plt.show()

# Corregir:
# - Hacer dataframe con todos los datos del pkl.
# - Hacer intersección entre la lista de pve_keys y el dataframe creado para solo tener los frames que se utilizaron en la calibración
# - Luego obtener df_rms y plotear con df_vel

# img0 = cv.imread(f'./sets/{args.file}Finf/{ret_names[rp0]}.jpg')
# displayImageWPoints(img0, proy_points_2D, real_points_2D, name=ffname)

# what's left
# determinar que es ese valor RMS (ojala error en puntos)
# correr los videos del dron y determinar periodos de velocidad angular alto

# hacer más calibraciones guardando rvec y tvec
# usar esos rvec y tvec para hacer reproyeccion de puntos
# calcular error rms promedio de la imagen


