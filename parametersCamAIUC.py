"""
Camera calibration parameters for Sony RX0-II owned by AIUC with their respective errors
"""
import numpy as np

# Means
fx = 2569.605957
fy = 2568.584961
cx = 1881.56543
cy = 1087.135376

camera_matrix = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]], dtype=np.float32)

k1 = 0.019473
k2 = -0.041976
p1 = -0.000273
p2 = -0.001083
k3 = 0.030603

dist_coeff = np.array(([k1], [k2], [p1], [p2], [k3]), dtype=np.float32)

# Sigmas
err_fx = 8.643641
err_fy = 8.33402
err_cx = 8.614324
err_cy = 3.526682

err_k1 = 0.005685
err_k2 = 0.023181
err_p1 = 0.000458
err_p2 = 0.000971
err_k3 = 0.027059