#!/usr/bin/env python

import os
import argparse
import cv2 as cv
import numpy as np
from tqdm import tqdm


def cb_balance(img, alpha=1.0, beta=0.0, auto=False):
    # Adjust contrast and brightness balance of the image
    if auto:
        alpha = 255 / (np.max(img)-np.min(img))
    if beta == 0 or auto:
        beta = -np.min(img)*alpha
    return alpha * img + beta


# Main function
def main():
    print(f'Getting frames from {args.vidname}')
    
    # Get name of the file only
    vidfile = os.path.basename(args.vidname)[:-4]
    
    # Open the video
    vidcap = cv.VideoCapture(args.vidname)
    
    # Get total frame count
    total_frame_count = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))
    
    # Initialize progress bar
    pbar = tqdm(desc='READING FRAMES', total=total_frame_count, unit=' frames', dynamic_ncols=True)
    frame_no = args.startnumber
    
    # Create output folders if they weren't created yet
    frames_path = os.path.normpath(os.path.dirname(args.vidname))+'/'+vidfile
    if not os.path.exists(frames_path):
        os.mkdir(frames_path)
    
    # Download all frames
    while(vidcap.isOpened()):
        frame_exists, curr_frame = vidcap.read()
        if frame_exists:
            if frame_no % args.reduction == args.residue:
                if args.alpha != 1.0 or args.beta != 0.0:
                    curr_frame = cb_balance(curr_frame, args.alpha, args.beta)
                cv.imwrite(f"{frames_path}/frame{frame_no}.jpg", curr_frame)
        else:
            pbar.close()
            print(f'All frames were saved in {frames_path}')
            break
        frame_no += 1
        pbar.update(1)
        
    # Release the file
    vidcap.release()
        

if __name__ == '__main__': 
    # Initialize parser
    parser = argparse.ArgumentParser(description='Extracts frames from a specified video.')
    parser.add_argument('vidname', type=str, help='Directory of video (mp4 format).')
    parser.add_argument('-o', '--output', type=str, metavar='folder', default='', help='Name of the folder that will contain the frames. If it is empty, the name of the file will be considered instead.')
    parser.add_argument('-rd', '--reduction', type=float, metavar='N', default=1, help='Reduction of number of frames (total/N).')
    parser.add_argument('-rs', '--residue', type=float, metavar='N', default=0, help='Residue or offset for the reduced number of frames.')
    parser.add_argument('-sn', '--startnumber', type=int, default=0, help="Number associated with the first frame and from where the count is starting. eg: 'frame0', 'frame1250'.")
    parser.add_argument('-af', '--alpha', type=float, default=1.0, help="Alpha value for change of Contrast of the images.")
    parser.add_argument('-bt', '--beta', type=float, default=0.0, help="Beta value for change of Contrast and Brightness of the images.")
    
    # Get parser arguments
    args = parser.parse_args()
    
    # Main
    main()
