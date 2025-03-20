#!/usr/bin/env python

import os
import argparse
import cv2 as cv
import numpy as np
import pandas as pd
import pymap3d as pm
import camera
from scipy.spatial import transform
from matplotlib import pyplot as plt


# Auxiliar function to plot the 3D space of the points from GSI with a simulated projection of those points using known intrinsic and extrinsic parameters of the camera.     
def plot_vectors(**kwargs):
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(121, projection='3d')
    try:
        pts3D = kwargs.get('p3D')
        cam = kwargs.get('camera')
        tgt = kwargs.get('target')
        ax.scatter(pts3D[:,0], pts3D[:,1], pts3D[:,2], zdir='z')
        ax.scatter(cam[0], cam[1], cam[2], zdir='z')
        ax.scatter(tgt[0], tgt[1], tgt[2], zdir='z')
        ax.quiver(cam[0], cam[1], cam[2], -1000*(cam[0]-tgt[0])/np.linalg.norm(cam-tgt), -1000*(cam[1]-tgt[1])/np.linalg.norm(cam-tgt), -1000*(cam[2]-tgt[2])/np.linalg.norm(cam-tgt))
    except:
        print("3D data can't be plotted")
    ax2 = fig.add_subplot(122)
    try:
        # Issue: projected points look mirrored but that's normal. Solution: fixed by inverting the value of Y axis (images goes top to bottom, while graph goes bottom to top)
        pts2D = kwargs.get('p2D')
        ax2.scatter(-pts2D[:,0,0], pts2D[:,0,1])
    except:
        print("2D data can't be plotted")

# Auxiliar function to calculate the RMSE between 2D points and simulated 2D points given intrinsic and extrinsic parameters of the camera.        
def projectionError(points_3D, points_2D, rvec, tvec, camera_matrix, dist_coeff):
    proj = cv.projectPoints(points_3D, rvec, tvec, camera_matrix, dist_coeff)[0]
    proj = proj.reshape(proj.shape[0], proj.shape[-1])
    res = proj - points_2D.reshape(points_2D.shape[0], points_2D.shape[-1])

    rms = np.sqrt(np.sum(res**2)/len(res))
    return rms

# Main function
def main():
    # Choose between GSI data (used for calibration), GPS data (from campaign site) or pseudo random generated data for 3D points
    if args.calibration_data:
        # Set filepath
        file_path = args.calibration_data
        
        # Check if file is csv or txt (old save vs new save)
        if os.path.splitext(file_path) == 'csv':
            # Read .csv and save it as a dataframe
            gpsArr = pd.read_csv(file_path)[['X','Y','Z']]
        else: # for txt files without header
            # Read .txt and save it as a dataframe
            header = ['X', 'Y', 'Z', 'sigmaX', 'sigmaY', 'sigmaZ', 'offset']
            gpsArr = pd.read_csv(file_path, delimiter='\t', names=header, index_col=0)[['X','Y','Z']]
        
        # Remove NUGGETS and CSB targets
        flt = gpsArr.index.str.contains("NUGGET|CSB|CODE133|CODE134|CODE135|CODE136|CODE137|CODE138")
        gpsArr = gpsArr[~flt]
        
        # Point of interest
        if args.poi:
            POI = gpsArr.loc[[args.poi]].to_numpy()[0]
        else:
            POI = gpsArr.loc[:, ['X', 'Y', 'Z']].mean().to_numpy()
        
        points_3D = gpsArr.to_numpy()
        points_3D -= POI
        
        gps_idx = gpsArr.index.tolist()

        # Plot the positions (X,Y) as seen from Front
        plt.figure(figsize=(30, 18))
        plt.scatter(points_3D[:,0], points_3D[:,1])
        
        # Add text to name every point in the image
        for i, name in enumerate(gps_idx):
            plt.text(points_3D[:,0][i], points_3D[:,1][i], name, fontsize=9, ha='right')
        
        # Save plot in Desktop
        plt.savefig(f'{os.path.join(os.path.expanduser("~"), "Desktop")}/{os.path.basename(file_path)[:-4]}.png', dpi=300)
        plt.close()
    
    elif args.gps_data:
        # Set filepath
        file_path = args.gps_data
        
        # Import the 3D points from a csv file containing GPS data (latitude, longitude, altitude)
        gpsArr = pd.read_csv(file_path, header=None, names=['Lat','Lon','Alt','Cluster','null'])
        gpsArr.dropna(axis=1, inplace=True)

        points_3D = gpsArr.to_numpy() # BEWARE: to_numpy() doesn't generate a copy but another instance to access the same data. So, if points_3D changes, gpsArr will too.
        
        # Point of interest (center)
        if args.poi:
            POI = gpsArr.loc[[args.poi]].to_numpy()[0]
        else:
            POI = gpsArr.loc[:, ['Lat', 'Lon', 'Alt']].mean().to_numpy()

        # Convert GPS data to a cartesian plane (local based)
        pts3D = []
        for i in range(points_3D.shape[0]):
            pnts = pm.geodetic2enu(points_3D[i,0], points_3D[i,1], points_3D[i,2], POI[0], POI[1], POI[2])
            pts3D.append(pnts)
        df_pts3D = pd.DataFrame(data=np.array(pts3D, dtype=np.float64), index=gpsArr.index.to_list(), columns=['X', 'Y', 'Z'])

        points_3D = df_pts3D.to_numpy()
        gps_idx = df_pts3D.index.to_list()
    
    else:
        file_path = 'random_set' + '    '
        # Set the same random seed for all experiments
        np.random.seed(0)
    
        # Generate a pseudo random dataset for 3D points
        N = 40
        plot2d = 100*np.random.randint(-20, 20, size=(N,2))
        plot3d = np.hstack((plot2d, np.zeros((plot2d.shape[0],1))))
        POI = np.array([0, 0, 0])
        
        points_3D = plot3d - POI
        gps_idx = [f'TARGET{i}' for i in range(N)]
        
    # Plot the positions (X,Y) as seen from Front
    plt.figure(figsize=(30, 18))
    plt.scatter(points_3D[:,0], points_3D[:,1])
    
    # Add text to name every point in the image
    for i, name in enumerate(gps_idx):
        plt.text(points_3D[:,0][i], points_3D[:,1][i], name, fontsize=9, ha='right')
    
    # Save plot in Desktop
    plt.savefig(f'{os.path.join(os.path.expanduser("~"), "Desktop")}/{os.path.basename(file_path)[:-4]}.png', dpi=300)
    plt.close()
    
    # Camera rotation
    ang = np.array(args.rotation)
    cam_r = transform.Rotation.from_euler('XYZ', ang, degrees=True) # 'XYZ' means intrinsic rotations (with respect of the same point/object and not a global reference) 
    rvec = cam_r.as_rotvec()

    # Camera position
    cam_t = np.array([0.0, 0.0, args.translation])
    tvec = - np.dot(cam_r.as_matrix(), cam_t)

    if args.calibfile:
        cam = camera.Camera(args.calib)
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

    # Simulate a picture of the camera using camera matrix and array of distortion coefficients
    points_2D = cv.projectPoints(points_3D, rvec, tvec, camera_matrix, dist_coeff)[0]
    
    # Plot 3D points, camera position and rotation and 2D points
    plot_vectors(p3D=points_3D, p2D=points_2D, camera=tvec, target=np.array([0,0,0]))
    # plt.savefig(f'camera_projections_sim/frame({i}-{j}-{k})-({ang[0]})({ang[1]})({ang[2]}).png')
    plt.show()
    plt.close()

    ## Reprojection check
    # Try to make a match of both to check if the solution is in itself consistent
    res, rvec0, tvec0 = cv.solvePnP(points_3D, points_2D, camera_matrix, dist_coeff)

    # Convert the resultant extrinsic rotation vector/matrix back to angles
    ang = cam_r.as_quat(canonical=True)
    r0 = transform.Rotation.from_rotvec(rvec0.flatten())
    ang0 = r0.as_quat(canonical=True)
    
    print(f'Reconstructed Angles with SolvePnP: {np.array2string(ang0, precision=2)} in quat')
    print(f'Original Angles:                    {np.array2string(ang, precision=2)} in quat')


if __name__ == '__main__': 
    # Initialize parser
    parser = argparse.ArgumentParser(description='Converts 3D points to 2D points by projecting them.')
    parser.add_argument('-gsi', '--calibration_data', type=str, default=False, help='Enables use of lab data (calibration process).')
    parser.add_argument('-site', '--gps_data', type=str, default=False, help='Enables use of gps data from campaign site.')
    parser.add_argument('-poi', type=str, default=False, help='Get an specific point as central point of the group (Default: the average position of the points).')
    parser.add_argument('-cb', '--calibfile', type=str, metavar='file', default=None, help='File directory containing calibration parameters (*.txt), for point reprojection and/or initial guess during calibration.')
    parser.add_argument('-r', '--rotation', type=float, nargs=3, metavar='N', default=[0, 180, 0], help='Rotation vector values (in degrees).')
    parser.add_argument('-t', '--translation', type=float, metavar='N', default=3300, help='Translation in axis Z (of the camera) from the point of interest.')

    # Get parser arguments
    args = parser.parse_args()
    
    # Main
    main()