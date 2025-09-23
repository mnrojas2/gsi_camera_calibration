import sys
import numpy as np
import pickle

def load_pkl(pkl_dir):
    pFile = pickle.load(open(pkl_dir, "rb"))

    # Unpack lists from the .pkl file(s)
    objpoints = pFile['3D_points']
    imgpoints = pFile['2D_points']
    ret_names = pFile['name_points']
    tgt_names = pFile['name_targets']
    vecs = pFile['rt_vectors']

    cam_mtx = pFile['init_mtx']
    dist_coeff = pFile['init_dist']
    img_shape = pFile['img_shape']
    init_calibfile = pFile['init_calibfile']

    return objpoints, imgpoints, ret_names, tgt_names, vecs, cam_mtx, dist_coeff, img_shape, init_calibfile

try:
    pkl0_dir = sys.argv[1] # 'C0157_vidpoints_15_normal.pkl'
    pkl1_dir = sys.argv[2] # 'C0157_vidpoints_15_from5to10.pkl'
except:
    print("Two directories for pkl files are needed")
    sys.exit(1)

p0_3d, p0_2d, p0_names, p0_targets, p0_vecs, p0_mtx, p0_dcoeff, p0_img, p0_calib = load_pkl(pkl0_dir)
p1_3d, p1_2d, p1_names, p1_targets, p1_vecs, p1_mtx, p1_dcoeff, p1_img, p1_calib = load_pkl(pkl1_dir)

# Get lengths of vectors (should be equal to number of frames read, minus 1)
print(len(p0_3d), len(p1_3d))

# Get length of 3D points (should be equal to number of points found in the image) of the first frame
print(p0_3d[0].shape, p1_3d[0].shape)
print(p0_2d[0].shape, p1_2d[0].shape)
print(p0_names[0], p1_names[0])
print(len(p0_targets[0]), len(p1_targets[0]))
print(p0_vecs[0].shape, p1_vecs[0].shape)

# Compare parameters shared by all frames
print("CAMERA MATRIX: ", np.array_equal(p0_mtx, p1_mtx))
print("DISTORTION COEFFICIENTS: ", np.array_equal(p0_dcoeff, p1_dcoeff))
print("IMAGE SHAPE: ", np.array_equal(p0_img, p1_img))
print("INITIAL CALIBRATION FILE: ", p0_calib == p1_calib)
input()

for i in range(len(p0_3d)):
    print(f"##### FRAME {i+1} ##### {p0_names[i]}")
    # Compare 3d points of each frame between all three datasets (should be the same thing)
    print("3D POINTS: ", np.array_equal(p0_3d[i], p1_3d[i]))

    # Compare 2d points of each frame between all three datasets (should be the same thing)
    print("2D POINTS: ", np.array_equal(p0_2d[i], p1_2d[i]))

    # Compare names of each frame between all three datasets (should be the same name)
    print("NAME: ", p0_names[i] == p1_names[i])

    # Compare target names of each frame between all three datasets (should be the same thing, in the same order)
    print("TARGET: ", p0_targets[i] == p1_targets[i])

    # Compare target names of each frame between all three datasets (should be the same thing, in the same order)
    print("ROTATION AND TRASLATION VECTORS: ", np.array_equal(p0_vecs[i], p1_vecs[i]))
    input()

