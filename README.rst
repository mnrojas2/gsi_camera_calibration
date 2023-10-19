==================
gsi_calibrationcam
==================

Projects points from GSI using OpenCV functions

Code summary
============

1) get_frames.py: Reads a .mp4 video and outputs a folder with frames.

#) point_tracking.py : Takes a folder with frames and outputs the calibration camera parameters using a new method to work with the custom board made in laboratory.

#) chessboard_cc.py: Takes a folder with frames and outputs the calibration camera parameters using the chessboard method.

#) circleboard_cc.py: Takes a folder with frames and outputs the calibration camera parameters using the circle grid method.

#) tabulate_yml_files.py: By reading a list of yml files containing camera calibration parameters, it sorts them and saves it in a .xlsx file.

#) points_detection.py: Detects calibration target points from a set of static images.

#) camera_sim_projection.py: Converts 3D points to 2D points by projecting them and compares the rotation and translation vectors after reconstruction. 

#) homography.py: Finds calibration target points by name by comparing codetarget points (found by hand) with the apparent location of the image in the 3D plot. Similar to point_tracking.py but it works for set C99Finf only.

#) just_calibration.py: Gets the calibration algorithm by loading a dataset in .pkl. Useful as it does not need to find the points in all frames more than once (done by running point_tracking.py).