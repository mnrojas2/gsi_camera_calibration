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

class cam_params(object):
    """Generic camera parameter class"""
    def __init__(self, cm, dist):
        self.cm = cm
        self.dist = dist

    def cam_matrix(self):
        return self.cm

    def dist_coeff(self):
        return self.dist

def read_param_file(filename):
    """
    Reads a parameter text file containing python executable entries.
    :param filename: text file
    :return: dictionary
    """
    import numpy as np
    # Do recursion and string substitution?
    di = {}
    # Load file.
    f = open(filename)
    line_num = 0
    exp = None
    for line in f:
        line_num += 1
        line = line.strip()
        s = line.split('#')
        if len(s) == 0 or len(s[0]) == 0:
            continue
        s = s[0]
        cont = s.strip().endswith('\\')
        if cont:
            s = s.split('\\')[0]
        if exp is None:
            exp = s
            exp_line = line_num
        else:
            exp = exp + s
        # Forced continuation?
        if cont:
            continue
        # Attempt evaluation
        if not '=' in exp:
            continue
        idx = exp.find('=')
        k, v = exp[:idx].strip(), exp[idx+1:].strip()
        try:
            v = eval(v)
        except NameError:
            print('name error in key %s, replacing nans...' % k)
            v = eval(v.replace('nan', 'float(\'nan\')'))
        except SyntaxError:
            # Any multi-line expression will get here at some point
            #print('syntax error in key %s' % k)
            continue
        di[k] = v
        exp = None
    if exp is not None:
        raise RuntimeError( \
            'Error while parsing %s -- incomplete or invalid eval block '\
            'starting on line %i' % (filename, exp_line))
    f.close()
    return di

def get_camera_params(paramfile):
    """
    Generic function to read camera parameters and return a camera parameter class.
    """
    # fx, fy, cx, cy, k1, k2, p1, p2, k3 = [0,0,0,0,0,0,0,0,0]
    camdata = read_param_file(paramfile)
    globals().update(camdata)
    cm = np.array([[fx, 0., cx],
                   [0., fy, cy],
                   [0., 0., 1.]])
    dist = np.array(([k1], [k2], [p1], [p2], [k3]))
    cam = cam_params(cm, dist)
    return cam

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
def gsi_projection(data, calibfile, link_points=[], **kwargs):
    # Read .txt and load it as a dataframe
    header = ['X', 'Y', 'Z', 'sigmaX', 'sigmaY', 'sigmaZ', 'offset']
    gpsArr = pd.read_csv(data, delimiter='\t', names=header, index_col=0)[['X','Y','Z']]
    
    # Remove NUGGETS and CSB targets
    flt = gpsArr.index.str.contains("NUGGET|CSB|CODE133|CODE134|CODE135|CODE136|CODE137|CODE138")
    gpsArr = gpsArr[~flt]
    
    # Point of interest
    poi_exists = kwargs.get('poi', False)
    if poi_exists:
        POI = gpsArr.loc[[poi_exists]].to_numpy()[0]
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
        if name in link_points:
            plt.scatter(points_3D[:,0][i], points_3D[:,1][i], color='red')
            plt.text(points_3D[:,0][i], points_3D[:,1][i], name, fontsize=9, ha='right')
    
    # Save plot in Desktop
    plt.savefig(f'{os.path.dirname(data)}/{os.path.basename(data)[:-4]}.png', dpi=300)
    plt.close()
    
    # Camera rotation
    rotation = kwargs.get('rotation', [0,180,0])
    ang = np.array(rotation)
    cam_r = transform.Rotation.from_euler('XYZ', ang, degrees=True) # 'XYZ' means intrinsic rotations (with respect of the same point/object and not a global reference) 
    rvec = cam_r.as_rotvec()

    # Camera position
    translation = kwargs.get('translation', 3300)
    cam_t = np.array([0.0, 0.0, translation])
    tvec = - np.dot(cam_r.as_matrix(), cam_t)

    # Load camera parameters
    cam = get_camera_params(calibfile)
    camera_matrix = cam.cam_matrix()
    dist_coeff = cam.dist_coeff()

    # Simulate a picture of the camera using camera matrix and array of distortion coefficients
    print(rvec, tvec, camera_matrix, dist_coeff)
    points_2D = cv.projectPoints(points_3D, rvec, tvec, camera_matrix, dist_coeff)[0]

    # Plot 3D points, camera position and rotation and 2D points
    plot_vectors(p3D=points_3D, p2D=points_2D, camera=tvec, target=np.array([0,0,0]))
    # plt.savefig(f'camera_projections_sim/frame({i}-{j}-{k})-({ang[0]})({ang[1]})({ang[2]}).png')
    plt.show()
    # plt.close()

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
    parser.add_argument('data', type=str, help='Path of a file containing points in a 3D environment (defined as x,y,z).')
    parser.add_argument('calibfile', type=str, metavar='file', help='File directory containing calibration parameters (*.txt), for point reprojection and/or initial guess during calibration.')

    parser.add_argument('-poi', type=str, default=None, help='Get an specific point as central point of the group (Default: the average position of the points).')
    parser.add_argument('-r', '--rotation', type=float, nargs=3, metavar='N', default=None, help='Rotation vector values (in degrees).')
    parser.add_argument('-t', '--translation', type=float, metavar='N', default=None, help='Translation in axis Z (of the camera) from the point of interest.')

    # Get parser arguments
    args = parser.parse_args()

    keys = ['poi', 'rotation', 'translation']
    kwargs = {k: getattr(args, k) for k in keys if getattr(args, k) is not None}
    
    # Define list of link_points (points that are considered important for projection-reprojection)
    link_targets = ["CODE131", "CODE139", "CODE140", "CODE141", "CODE142", "CODE143", "CODE144",
                    "TARGET246", "TARGET295", "TARGET176", "TARGET142", "TARGET116", "TARGET286", 
                    "TARGET3", "TARGET313", "TARGET321"]
    
    # Main
    gsi_projection(args.data, args.calibfile, link_targets, **kwargs)