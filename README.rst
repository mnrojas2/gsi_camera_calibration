==================
gsi_calibrationcam
==================

Projects points from GSI using OpenCV functions

Code summary
============

1) get_frames.py: Reads a .mp4 video and outputs a folder with frames.

#) chessboard_cc.py: Takes a folder with frames and outputs the calibration camera parameters using the chessboard method.

#) circleboard_cc.py: Takes a folder with frames and outputs the calibration camera parameters using the circle grid method.

#) read_yml_files.py: By reading a list of yml files containing camera calibration parameters, it tabulates them and saves it in a .xlsx file.

#) points_detection.py: Detects calibration target points from a set of images.

#) coords_camera.py: Converts 3D points to 2D points by projecting them.

#) homography_tests.py: Finds calibration target points by name by comparing codetarget points (found by hand) with the apparent location of the image in the 3D plot.