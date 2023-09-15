import numpy as np
import pandas as pd
import cv2
from scipy.spatial.transform import Rotation as R
from astropy import units as u
from tqdm import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
      
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
        ax.quiver(cam[0], cam[1], cam[2], -50*(cam[0]-tgt[0])/np.linalg.norm(cam-tgt), -50*(cam[1]-tgt[1])/np.linalg.norm(cam-tgt), -50*(cam[2]-tgt[2])/np.linalg.norm(cam-tgt))
    except:
        print("3D data can't be plotted")
    ax2 = fig.add_subplot(122)
    try:
        pts2D = kwargs.get('p2D')
        ax2.scatter(pts2D[:,0,0], pts2D[:,0,1])
    except:
        print("2D data can't be plotted")
            
def projectionError(points_3D, points_2D, rvec, tvec, camera_matrix, dist_coeff):
    proj = cv2.projectPoints(points_3D, rvec, tvec, camera_matrix, dist_coeff)[0]
    proj = proj.reshape(proj.shape[0], proj.shape[-1])
    res = proj - points_2D.reshape(points_2D.shape[0], points_2D.shape[-1])

    rms = np.sqrt(np.sum(res**2)/len(res))

    return rms

np.random.seed(0)

def main():
    real_data = True # True
    if real_data:
        gpsArr = pd.read_csv('./videos/Coordenadas/Bundle2.csv')[['X','Y','Z']]
        
        # # Point of interest
        POI = gpsArr.loc[['CODE45']].to_numpy()[0] # all
        # POIsp = gpsArr.loc[['CODE46']].to_numpy()[0] # small panel
        # POItb = gpsArr.loc[['TARGET311']].to_numpy()[0] # table
        # POIwl = gpsArr.loc[['TARGET317']].to_numpy()[0] # wall
        # POIbp = gpsArr.loc[['CODE32']].to_numpy()[0] # big panel
        
        # ajustar escala a algo cercano a las distancias en cm
        points_3D = gpsArr.to_numpy() #2.545*
        points_3D -= POI
    
    else:
        N = 40
        plot2d = np.random.randint(-20, 20, size=(N,2))
        plot3d = np.hstack((plot2d, np.zeros((plot2d.shape[0],1))))
        POI = np.array([0, 0, 0])
        points_3D = plot3d - POI

    # for i in tqdm(range(18+1)):
    #     for j in tqdm(range(18+1), leave=False):
    #         for k in tqdm(range(18+1), leave=False):
    #             ang = np.array((-45+5*i, 135+5*j, -45+5*k))
    
    # Camera rotation
    ang = np.array((0, 180, 0))
    cam_r = R.from_euler('XYZ', ang, degrees=True)
    rvec = cam_r.as_rotvec()

    # Camera position
    cam_t = np.array((0, 0, 200)) # sim: 50 # irl points: 
    tvec = - np.dot(cam_r.as_matrix(), cam_t)

    # Camera matrix parameters
    fx = 2605.170124
    fy = 2596.136808
    cx = 1882.683683
    cy = 1072.920820

    camera_matrix = np.array([[fx, 0., cx],
                            [0., fy, cy],
                            [0., 0., 1.]], dtype = "double")

    # Camera distortion coefficients parameters
    k1 = -0.02247760754865920 # -0.011935952
    k2 =  0.48088686640699946 #  0.03064728
    p1 = -0.00483894784615441 # -0.00067055
    p2 =  0.00201310943773827 # -0.00512621
    k3 = -0.38064382293946231 # -0.11974069
    # k4 = 
    # k5 = 
    # k6 =

    dist_coeff = np.array(([k1], [k2], [p1], [p2], [k3])) # [k4], [k5], [k6]))
    # dist_coeff = np.zeros(5, np.float32)

    # Simulate a picture of the camera using camera matrix and array of distortion coefficients
    #   projected points look mirrored but that's normal
    points_2D = cv2.projectPoints(points_3D, rvec, tvec, camera_matrix, dist_coeff)[0]
    
    # Plot 3D points, camera position and rotation and 2D points
    plot_vectors(p3D=points_3D, p2D=points_2D, camera=tvec, target=np.array([0,0,0]))
    # plt.savefig(f'camera_projections_sim/frame({i}-{j}-{k})-({ang[0]})({ang[1]})({ang[2]}).png')
    # plt.close()
    plt.show()

    # Try to make a match of both to check if the solution is in itself consistent
    res, rvec0, tvec0 = cv2.solvePnP(points_3D, points_2D, camera_matrix, dist_coeff)

    # Convert the resultant  extrinsic rotation vector/matrix back to angles
    ang = cam_r.as_quat(canonical=True)
    r0 = R.from_rotvec(rvec0.flatten())
    ang0 = r0.as_quat(canonical=True)
    
    print(f'Reconstructed Angles with SolvePnP: {np.array2string(ang0, precision=2)} in quat')
    print(f'Original Angles: {np.array2string(ang, precision=2)} in quat')
        
if __name__ == '__main__': main()