#!/usr/bin/env python

import os
import argparse
import pandas as pd
import numpy as np
import cv2 as cv
import pickle
import copy
from scipy.spatial.transform import Rotation as R 

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
            clr += [-128, 128, 128]
            clr = (np.array(clr) + np.random.randint(-128, 128, size=3)).tolist()
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

# Initialize parser
parser = argparse.ArgumentParser(description='Camera calibration using chessboard images.')
parser.add_argument('file', type=str, help='Name of the file containing data (*pkl).')

###############################################################################################################################
# Main

# Get parser arguments
args = parser.parse_args()

# Load pickle file
print(f'Loading {args.file}.pkl')
pFile = pickle.load(open(f"./datasets/pkl-files/{args.file}.pkl","rb"))

# Unpack lists
objpoints = pFile['3D_points']
imgpoints = pFile['2D_points']
ret_names = pFile['name_points']
tgt_names = pFile['name_targets']

camera_matrix = pFile['init_mtx']
dist_coeff = pFile['init_dist']
img_shape = pFile['img_shape']

calibfile = pFile['init_calibfile']
vecs = pFile['rt_vectors']

tgt0 = 1250
tgt1 = 570

tg0 = pd.DataFrame(data=imgpoints[tgt0].reshape(-1, 2), index=tgt_names[tgt0], columns=['X', 'Y'])
img0 = cv.imread(f'./sets/{args.file[:-10]}/{ret_names[tgt0]}.jpg')
# displayImageWPoints(img0, tg0, name=ret_names[tgt0], show_names=True)

tg1 = pd.DataFrame(data=imgpoints[tgt1].reshape(-1, 2), index=tgt_names[tgt1], columns=['X', 'Y'])
img1 = cv.imread(f'./sets/{args.file[:-10]}/{ret_names[tgt1]}.jpg')
displayImageWPoints(img1, tg1, name=ret_names[tgt1], show_names=True)

tgt_common = (tg0.index.intersection(tg1.index)).tolist()
# displayImageWPoints(img0, tg0.loc[tgt_common], name=ret_names[tgt0], show_names=True)
displayImageWPoints(img1, tg1.loc[tgt_common], name=ret_names[tgt1], show_names=True)

# what's left
## get the distance between both points
## divide by the time between frames (1/29.97)
## put them on a list to plot vel vs rmse



