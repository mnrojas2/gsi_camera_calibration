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
from sklearn.linear_model import LinearRegression

# Initialize parser
parser = argparse.ArgumentParser(description='Camera calibration using chessboard images.')
parser.add_argument('file', type=str, help='Name of the file containing data (*.pkl).')
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
print(f'Loading {args.file}Finf_vidpoints.pkl')
pFile = pickle.load(open(f"./datasets/pkl-files/{args.file}Finf_vidpoints.pkl","rb"))

# Unpack lists
objpoints = pFile['3D_points']
imgpoints = pFile['2D_points']
ret_names = pFile['name_points']
tgt_names = pFile['name_targets']

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

# Load .yml file
print(f'Loading {args.file}-{args.calibfile}.yml')
fs = cv.FileStorage(f'./results/{args.file}-{args.calibfile}.yml', cv.FILE_STORAGE_READ)
print(f"File '{args.file}-{args.calibfile}.yml' description:",fs.getNode("summary").string())

camera_matrix = fs.getNode("camera_matrix").mat() # Aquí estaba el problema, no se había inicializado con el nombre correcto y estaba tomando el valor que venía en el pkl.
dist_coeff = fs.getNode("dist_coeff").mat()

#############################################################################################

#'''
# Calculating RMS Error by hand (using rvec and tvec from file)
# UPDATE: using camera_matrix and dist_coeff, we can reconstruct the same rvec and tvec vectors that are saves in the calibration file. In that way, we can confirm we can make them ourselves to get more values.
rms_error = []
rms_names = []

for i in range(len(imgpoints)):
    ffname = ret_names[i]
    real_points_2D = imgpoints[i]

    # Project points
    _, rvec, tvec = cv.solvePnP(objectPoints=objpoints[i], imagePoints=real_points_2D, cameraMatrix=camera_matrix, distCoeffs=dist_coeff)
    proy_points_2D = cv.projectPoints(objectPoints=objpoints[i], rvec=rvec, tvec=tvec, cameraMatrix=camera_matrix, distCoeffs=dist_coeff)[0]
    
    # Calculate the norm between real and proyected points, then calculate the RMS error
    dist_pts2D = cv.norm(proy_points_2D.reshape(-1,2), real_points_2D.reshape(-1,2), normType=cv.NORM_L2)
    mean_pts2D = np.sqrt(np.dot(dist_pts2D, dist_pts2D)/real_points_2D.shape[0])
    
    rms_error.append(mean_pts2D)
    rms_names.append(ffname)
    
df_rms = pd.DataFrame(data=np.array(rms_error), index=rms_names, columns=['Error'])
rms_vel = (df_rms.index.intersection(df_vel.index)).tolist()

x_rms = df_rms.loc[rms_vel].to_numpy()
y_vel = df_vel.loc[rms_vel].to_numpy()/np.sqrt(camera_matrix[0,0]*camera_matrix[1,1]) * 180/np.pi # to convert to degrees/s

# Plot curves to check correlation
fig, ax1 = plt.subplots(figsize=(12, 7))

color = 'tab:red'
ax1.set_xlabel('frame (i)')
ax1.set_ylabel('RMS Error amplitude (pixels)')
ax1.plot(x_rms, color=color)
ax1.tick_params(axis='y', labelcolor=color)

color = 'tab:blue'
ax2 = ax1.twinx()
ax2.set_xlabel('frame (i)')
ax2.set_ylabel('Angular velocity (pixels/s)')
ax2.plot(y_vel, color=color)
ax2.tick_params(axis='y', labelcolor=color)
plt.title(f'Angular Velocity and RMS Error vs time (frames) ({fs.getNode("summary").string()})')
fig.tight_layout()
# plt.savefig(f'./plots/{args.file}-{args.calibfile}.jpg', bbox_inches='tight', dpi=300)

'''
# Hacer fit al scatter para determinar relación entre velocidad angular (en grados) con error rms. # Done
# Revisar que los valores estén bien (usar el valor calculado a mano de los RMS, ...
#   probar determinando rvec y tvec usando los valores definidos de camera_matrix, dist_coeff ...
#   para compararlo con los resultados del archivo) # Done
# Histograma -> revisar valores, sacar outliers, encontrar valores ideales ajustados por ajuste gaussiano. # Falta
# Determinar error en metros de los puntos (dron) # Ni idea cómo calcularlo
'''

# Linear regression
model = LinearRegression()
model.fit(x_rms, y_vel)
y_pred = model.predict(x_rms)
r_sq = model.score(x_rms, y_vel)
print(f"Coefficient of determination: {r_sq}")
print(f"Intercept (b0): {model.intercept_}, Slope (m): {model.coef_}")

plt.figure(figsize=(12, 7))
plt.scatter(x_rms, y_vel)
plt.plot(x_rms, y_pred, color='k')
plt.xlabel('RMS Error (pixels)')
plt.ylabel('Angular velocity (pixels/s)') # (degrees/s)')
plt.title('Angular velocity vs RMS Error')
plt.tight_layout()
plt.show() # '''

# determinar que es ese valor RMS (ojala error en puntos) -> Es el error rms promedio asociado a la distancia del punto en pixeles
# correr los videos del dron y determinar periodos de velocidad angular alto

# hacer más calibraciones guardando rvec y tvec
# usar esos rvec y tvec para hacer reproyeccion de puntos
# calcular error rms promedio de la imagen


