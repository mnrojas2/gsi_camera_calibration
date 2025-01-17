#!/usr/bin/env python

import cv2 as cv
import numpy as np
import os
import argparse
from tqdm import tqdm

# Initialize parser
parser = argparse.ArgumentParser(description='Extracts frames from a specified video.')
parser.add_argument('vidname', type=str, help='Name of video (mp4 format).')
parser.add_argument('-o', '--output', type=str, metavar='folder', default='', help='Name of the folder that will contain the frames. If it is empty, the name of the file will be considered instead.')
parser.add_argument('-rd', '--reduction', type=float, metavar='N', default=1, help='Reduction of number of frames (total/N).')
parser.add_argument('-rs', '--residue', type=float, metavar='N', default=0, help='Residue or offset for the reduced number of frames.')
parser.add_argument('-sn', '--startnumber', type=int, default=0, help="Number associated with the first frame and from where the count is starting. eg: 'frame0', 'frame1250'.")
parser.add_argument('-af', '--alpha', type=float, default=1.0, help="Alpha value for change of Contrast of the images.")
parser.add_argument('-bt', '--beta', type=float, default=0.0, help="Beta value for change of Contrast and Brightness of the images.")

def cb_balance(img, alpha=1.0, beta=0.0, auto=False):
    # Adjust contrast and brightness balance of the image
    if auto:
        alpha = 255 / (np.max(img)-np.min(img))
    if beta == 0 or auto:
        beta = -np.min(img)*alpha
    return alpha * img + beta

# Main function
def main():
    # Take all arguments from terminal
    args = parser.parse_args()
    print(f'Getting frames from {args.vidname}')
    
    try:
        # Start the video to take the necessary frames
        vidcap = cv.VideoCapture(args.vidname)
        total_frame_count = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))
        if total_frame_count == 0:
            # Since cv.VideoCapture can't force errors when no video is found, we do it manually.
            raise IndexError
        
        # Get name of the file (not directory) to use it as folder name
        if args.output == '':
            vid_name = (args.vidname).replace('\\', '/').split('/')[-1][:-4]
        else:
            vid_name = args.output
        
        # Create output folder if it wasn't created yet
        if not os.path.exists(f'sets/{vid_name}'):
            os.mkdir(f'sets/{vid_name}')
        
        # Start counters
        pbar = tqdm(desc='READING FRAMES', total=total_frame_count, unit=' frames', dynamic_ncols=True)
        frame_no = args.startnumber
        
        while(vidcap.isOpened()):
            frame_exists, curr_frame = vidcap.read()
            if frame_exists:
                if frame_no % args.reduction == args.residue:
                    if args.alpha != 1.0 or args.beta != 0.0:
                        curr_frame = cb_balance(curr_frame, args.alpha, args.beta)
                    cv.imwrite("sets/"+vid_name+"/frame%d.jpg" % frame_no, curr_frame)
            else:
                pbar.close()
                print(f'All frames were saved in /sets/{vid_name}')
                break
            frame_no += 1
            pbar.update(1)
        vidcap.release()
    except:
        print(f'No video was found in {args.vidname}')

if __name__ == '__main__': main()
