#!/usr/bin/env python

import argparse
import numpy as np
import pickle

# Initialize parser
parser = argparse.ArgumentParser(description='Script to remove points that are too far away from the main board.')
parser.add_argument('file', nargs='+', type=str, help='Name of the file containing calibration data (*.pkl).')


# Main
def main():
    # Get parser arguments
    args = parser.parse_args()

    # Load pkl file
    dirfile = args.file[0] # './datasets/pkl-files/old/C43Finf_vidpoints.pkl'
    pFile = pickle.load(open(dirfile, "rb"))
    print(f"Loading pkl file {dirfile}")

    # Unpack lists from the .pkl file(s)
    objpoints = pFile['3D_points']
    imgpoints = pFile['2D_points']
    ret_names = pFile['name_points']
    nam_targets = pFile['name_targets']

    camera_matrix = pFile['init_mtx']
    dist_coeff = pFile['init_dist']
    img_shape = pFile['img_shape']

    calibfile = pFile['init_calibfile']
    vecs = pFile['rt_vectors']

    # Filter points that are too far in the Z axis
    objpoints_new = []
    imgpoints_new = []

    for i in range(len(objpoints)):
        obj3D = objpoints[i]
        img2D = imgpoints[i]
        
        # Filter all points whose Z axis is over 400 in both lists
        obj3D_ftr = obj3D[obj3D[:,2] < 400]
        img2D_ftr = img2D[obj3D[:,2] < 400]
        
        # Append them in new lists to save
        objpoints_new.append(obj3D_ftr)
        imgpoints_new.append(img2D_ftr)

    # Save pkl file
    vid_data = {'3D_points': objpoints_new, '2D_points': imgpoints_new, 'name_points': ret_names, 'name_targets': nam_targets, 'rt_vectors': vecs,
                    'init_mtx': camera_matrix, 'init_dist': dist_coeff, 'img_shape': img_shape, 'init_calibfile': calibfile}
    with open(dirfile[:-4]+'_filtered.pkl', 'wb') as fp:
        pickle.dump(vid_data, fp)
        print(f"Dictionary successfully saved as '{dirfile[:-4]}_filtered.pkl'")

   
if __name__ == '__main__':
    main()