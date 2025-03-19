#!/usr/bin/env python

import os
import argparse
import pandas as pd
import numpy as np
import cv2 as cv
import pickle
import copy
import camera
from scipy.spatial.transform import Rotation as R 
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


# Functions
def displayImage(img, width=1280, height=720, name='Picture'):
    # Small simple function to display images without needing to add the auxiliar functions. By default it reduces the size of the image to 1280x720.
    cv.imshow(name, cv.resize(img, (width, height)))
    cv.waitKey(0)
    cv.destroyAllWindows()
        
def displayImageWPoints(img, *args, name='Image', show_names=False, save=False, fdir='new_set'):
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
        # Create output folder if it wasn't created yet
        ffolder, fvid = fdir.split('\\')
        if not os.path.exists(ffolder+'/tracked_sets/'):
            os.mkdir(ffolder+'/tracked_sets/')
        if not os.path.exists(ffolder+'/tracked_sets/'+fvid):
            os.mkdir(ffolder+'/tracked_sets/'+fvid)
        cv.imwrite(f'{ffolder}/tracked_sets/{fvid}/{name}.jpg', img_copy)
    else:
        displayImage(img_copy, name=name)



# Main
def main():
    # Load pickle file
    print(f'Loading {args.file}')
    pFile = pickle.load(open(f"{args.file}","rb"))

    # Unpack lists
    objpoints = pFile['3D_points']
    imgpoints = pFile['2D_points']
    ret_names = pFile['name_points']
    tgt_names = pFile['name_targets']

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

    # Load camera parameters from txt file or using values by default defined in this script
    if args.calibfile:
        cam = camera.Camera(args.calibfile)
        camera_matrix = cam.cam_matrix()
        dist_coeff = cam.dist_coeff()

    else:
        # Camera matrix
        fx = 2569.605957
        fy = 2568.584961
        cx = 1881.56543
        cy = 1087.135376

        camera_matrix = np.array([[fx, 0., cx],
                                    [0., fy, cy],
                                    [0., 0., 1.]], dtype = "double")

        # Distortion coefficients
        k1 = 0.019473
        k2 = -0.041976
        p1 = -0.000273
        p2 = -0.001083
        k3 = 0.030603

        dist_coeff = np.array(([k1], [k2], [p1], [p2], [k3]))
        summary = 'averaged values'

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

    color = 'tab:blue'
    ax1.set_xlabel('frame (i)')
    ax1.set_ylabel('Angular speed (degrees/s)')
    ax1.plot(y_vel, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    color = 'tab:red'
    ax2 = ax1.twinx()
    ax2.set_xlabel('frame (i)')
    ax2.set_ylabel('RMS Error amplitude (pixels)')
    ax2.plot(x_rms, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Angular speed and RMS Error vs time (frames)')
    fig.tight_layout()
    # plt.savefig(f'./plots/{args.file}-{args.calibfile}.jpg', bbox_inches='tight', dpi=300)

    if args.calibfile:
        # Plot difference in errors
        plt.figure(figsize=(12, 7))
        plt.plot(np.arange(len(rms_error)), rms_error, label='Calculated')
        plt.plot(np.arange(len(rms_error))[::20], rms_error[::20], label='Calculated 5%')
        plt.xlabel('Frames (i)')
        plt.ylabel('RMS Error (Pixels)')
        plt.legend()
        plt.title('RMS Error vs Time (frames)')

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
    plt.ylabel('Angular speed (degrees/s)') # (degrees/s)')
    plt.title('Angular speed vs RMS Error')
    plt.tight_layout()


    plt.figure(figsize=(12, 7))
    plt.hist(y_vel, bins=20, label='y_vel')
    plt.hist(x_rms*model.coef_+model.intercept_, bins=20, label='x_rms*m+b')
    plt.title('Histograms of Angular speed and RMS Error')
    plt.legend()
    plt.tight_layout()
    plt.show() # '''

    # determinar que es ese valor RMS (ojala error en puntos) -> Es el error rms promedio asociado a la distancia del punto en pixeles
    # correr los videos del dron y determinar periodos de velocidad angular alto

    # hacer más calibraciones guardando rvec y tvec
    # usar esos rvec y tvec para hacer reproyeccion de puntos
    # calcular error rms promedio de la imagen


if __name__ == '__main__':
    # Initialize parser
    parser = argparse.ArgumentParser(description='Calculates and correlates angular speed with RMS Error from a list targets.')
    parser.add_argument('file', type=str, help='Name of the file containing data (*.pkl).')
    parser.add_argument('-cb', '--calibfile', type=str, metavar='file', help='Name of the file containing calibration results (*.yml), for point reprojection and/or initial guess during calibration.')
    
    # Get parser arguments
    args = parser.parse_args()
    
    # Main
    main()