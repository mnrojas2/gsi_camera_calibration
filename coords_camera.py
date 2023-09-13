import numpy as np
import pandas as pd
import cv2
from scipy.spatial.transform import Rotation as R
from astropy import units as u
from tqdm import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
    
def plot_vectors(*args, d_points=False, vector=False):
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    for arg in args:
        try:
            ax.scatter(arg[:,0], arg[:,1], arg[:,2], zdir='z')
        except IndexError:
            ax.scatter(arg[0], arg[1], arg[2], zdir='z')
    ax2 = fig.add_subplot(122)
    if type(d_points) != bool:
        ax2.scatter(d_points[:,0,0], d_points[:,0,1])
    if type(vector) != bool:
        try:
            ax.quiver(vector[0], vector[1], vector[2], vector[3], vector[4], vector[5])
        except:
            print("Check your vector\n", vector)
            
def projectionError(points_3D, points_2D, rvec, tvec, camera_matrix, dist_coeff):

    proj = cv2.projectPoints(points_3D, rvec, tvec, camera_matrix, dist_coeff)[0]
    proj = proj.reshape(proj.shape[0], proj.shape[-1])
    res = proj - points_2D.reshape(points_2D.shape[0], points_2D.shape[-1])

    rms = np.sqrt(np.sum(res**2)/len(res))

    return rms

# gpsArr = pd.read_csv('./videos/Coordenadas/Bundle2.csv')[['X','Y','Z']]
# points_3D = gpsArr.to_numpy()

# # Point of interest
# POI_all = gpsArr.loc[['CODE45']].to_numpy()[0]
# POI_pnl = gpsArr.loc[['CODE46']].to_numpy()[0]
# POI_tbl = gpsArr.loc[['TARGET311']].to_numpy()[0]
# POI_wll = gpsArr.loc[['TARGET317']].to_numpy()[0]
# POI_bgpnl = gpsArr.loc[['CODE32']].to_numpy()[0]

np.random.seed(0)

N = 40
plot2d = np.random.randint(-20, 20, size=(N,2))
plot3d = np.hstack((plot2d, np.zeros((plot2d.shape[0],1))))
origin = np.array([0,0,0])#plot3d[int(N/2), :]
points_3D = plot3d - origin

# Camera rotation
ang = np.array((0, 180, 0))
cam_r = R.from_euler('xyz', ang, degrees=True)
rvec = cam_r.as_mrp()

# Camera position
cam_t = np.array((0, 0, 5000))
tvec = -np.dot(cam_r.as_matrix(), cam_t)

# Camera parameters matrix
fx = 2605.170124
fy = 2596.136808
cx = 1882.683683
cy = 1072.920820

camera_matrix = np.array([[fx, 0., cx],
                          [0., fy, cy],
                          [0., 0., 1.]], dtype = "double")

# Camera parameters distortion coefficients
k1 = -0.02247760754865920 # -0.011935952
k2 =  0.48088686640699946 #  0.03064728
p1 = -0.00483894784615441 # -0.00067055
p2 =  0.00201310943773827 # -0.00512621
k3 = -0.38064382293946231 # -0.11974069
# k4 = 
# k5 = 
# k6 = 

dist_coeff = np.array(([k1], [k2], [p1], [p2], [k3])) # [k4], [k5], [k6])) 
# dist_coeff = np.zeros((5, 1), np.float32)

# Simulate a picture of the camera using camera matrix and array of distortion coefficients
points_2D = cv2.projectPoints(points_3D, rvec, tvec, camera_matrix, dist_coeff)[0] #
plot_vectors(points_3D, tvec, d_points=points_2D)
plt.show()

# Try to make a match of both to check if the solution is in itself consistent
res, rvec0, tvec0 = cv2.solvePnP(points_3D, points_2D, camera_matrix, dist_coeff)

# Convert the resultant  extrinsic rotation vector/matrix back to angles
r = R.from_mrp(rvec0.flatten())
ang0 = r.as_euler('xyz', degrees=True)
ang = cam_r.as_quat(canonical=True)
ang0 = r.as_quat(canonical=True)

print(f'Reconstructed Angles with SolvePnP: {np.array2string(ang0, precision=2)} in quat')
print(f'Original Angles: {np.array2string(ang, precision=2)} in quat')