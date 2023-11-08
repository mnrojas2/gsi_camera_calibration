#!/usr/bin/env python

import cv2 as cv
import os
import argparse
from tqdm import tqdm

# Initialize parser
parser = argparse.ArgumentParser(description='Extracts frames from the specified video.')
parser.add_argument('vidname', type=str, help='Name of video (mp4 format).')
parser.add_argument('-in', '--folder', type=str, metavar='folder', dest='origin_folder', default='', help='Name of the folder/directory containing the video. Video or directory must be inside ./videos/ folder.')
parser.add_argument('-rd', '--reduction', type=float, metavar='N', default=1, help='Reduction of number of frames (total/N).')
parser.add_argument('-rs', '--residue', type=float, metavar='N', default=0, help='Residue or offset for the reduced number of frames.')
parser.add_argument('-sn', '--startnumber', type=int, default=0, help="Number associated with the first frame and from where the count is starting. eg: 'frame0', 'frame1250'.")

# Main function
def main():
    # Take all arguments from terminal
    args = parser.parse_args()
    print(f'Getting frames from ./videos/{args.origin_folder}/{args.vidname}.mp4')
    
    try:
        # Start the video to take the necessary frames
        vidcap = cv.VideoCapture('videos/'+args.origin_folder+'/'+args.vidname+'.mp4')
        total_frame_count = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))
        if total_frame_count == 0:
            # Since cv.VideoCapture can't force errors when no video is found, we do it manually.
            raise IndexError
        
        # Create output folder if it wasn't created yet
        if not os.path.exists('sets/'+args.vidname):
            os.mkdir('sets/'+args.vidname)
        
        # Start counters
        pbar = tqdm(desc='READING FRAMES', total=total_frame_count, unit=' frames', dynamic_ncols=True)
        frame_no = args.startnumber
        
        while(vidcap.isOpened()):
            frame_exists, curr_frame = vidcap.read()
            if frame_exists:
                if frame_no % args.reduction == args.residue:
                    cv.imwrite("sets/"+args.vidname+"/frame%d.jpg" % frame_no, curr_frame)
            else:
                pbar.close()
                print(f'All frames were saved in /sets/{args.vidname}')
                break
            frame_no += 1
            pbar.update(1)
        vidcap.release()
    except:
        print(f'No video named {args.vidname}.mp4 was found in ./videos/{args.origin_folder}/')

if __name__ == '__main__': main()
