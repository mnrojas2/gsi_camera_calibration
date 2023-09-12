import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from astropy import units as u
from tqdm import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)
    
def plot_vectors(*args):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for arg in args:
        try:
            ax.scatter(arg[:,0], arg[:,1], arg[:,2], zdir='z')
        except IndexError:
            ax.scatter(arg[0], arg[1], arg[2], zdir='z')
    plt.show()

gpsArr = pd.read_csv('./videos/Coordenadas/Bundle2.csv')[['X','Y','Z']]
gpsArr2 = gpsArr.to_numpy()

# Point of interest
POI_all = gpsArr.loc[['CODE45']].to_numpy()[0]
POI_pnl = gpsArr.loc[['CODE46']].to_numpy()[0]
POI_tbl = gpsArr.loc[['TARGET311']].to_numpy()[0]
POI_wll = gpsArr.loc[['TARGET317']].to_numpy()[0]
POI_bgpnl = gpsArr.loc[['CODE32']].to_numpy()[0]

# plot_points(POI_all, POI_pnl, POI_tbl, POI_wll, POI_bgpnl)

N = 20
plot2d = np.random.randint(-20, 20, size=(N,2))
plot3d = np.hstack((plot2d, np.zeros((plot2d.shape[0],1))))
origin = plot3d[int(N/2), :]
new_plot3d = plot3d - origin

# Camera parameters matrix
fx = 2617.155135
fy = 2618.742526
cx = 1905.942697
cy = 1086.927147

# Camera parameters distortion coefficients
k1 = -0.002660225 # -0.011935952
k2 =  0.175638082 #  0.03064728
p1 = -0.000600269 # -0.00067055
p2 =  0.003928041 # -0.00512621
k3 = -0.389710329 # -0.11974069
# k4 = 
# k5 = 
# k6 = 

camera_matrix = np.array([[fx, 0., cx],
                          [0., fy, cy],
                          [0., 0., 1.]], dtype = "double")

dist_coeff = np.array(([k1], [k2], [p1], [p2], [k3])) # [k4], [k5], [k6]))

# Camera rotation
cam_r = R.from_mrp([0, 0, 1])

# Camera position
cam_t = np.array((0, 0, 50))
tvec = - np.dot(cam_r.as_matrix(), cam_t)
plot_vectors(new_plot3d, cam_t)