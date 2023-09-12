import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

def main():
    # Take all arguments in the terminal
    args = sys.argv[1:]
    
    # First argument: name of the video to get frames
    vidname = args[0]
    
    # Second and third arguments: name of the folder containing the video (folder inside ./videos/)
    if len(args) >= 3 and args[1] in ('--folder','-in'):
        origin_folder = args[2]
        print('Searching video in: videos/' + origin_folder + '/' + vidname + '.mp4')
    else: 
        origin_folder = ''
    
    # Fourth and fifth arguments: reduction of number of frames (1/x)
    if len(args) >= 5 and args[3] in ('--reduction','-rd'):
        reduction = int(args[4])
    else: 
        reduction = 10
    
    # Fifth and sixth arguments: Residue or offset for the reduced number of frames 
    if len(args) >= 7 and args[5] in ('--residue', '-rs'):
        residue = int(args[6])
    else:
        residue = 0
    
    # Create output folder if it wasn't created yet
    if not os.path.exists('sets/'+vidname):
        os.mkdir('sets/'+vidname)
        
    # Start the video to take its frames after it's done finish all processes
    vidcap = cv2.VideoCapture('videos/' + origin_folder + '/' + vidname + '.mp4')
    total_frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(desc='READING FRAMES', total=total_frame_count, unit=' frames')
    frame_no = 0
    
    while(vidcap.isOpened()):
        frame_exists, curr_frame = vidcap.read()
        if frame_exists:
            if frame_no % reduction == residue:
                cv2.imwrite("sets/" + vidname + "/frame%d.jpg" % frame_no, curr_frame)
        else:
            pbar.close()
            print("All frames were saved in /sets/" + vidname)
            break
        frame_no += 1
        pbar.update(1)
    vidcap.release()

if __name__ == '__main__': main()
