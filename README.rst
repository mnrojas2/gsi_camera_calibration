==================
gsi_calibrationcam
==================

Projects points from GSI using OpenCV functions

Code summary
============

1) get_frames.py: Reads a .mp4 video and outputs a folder with frames.

#) point_tracking.py : Takes a folder with frames and outputs the calibration camera parameters using a new method to work with the custom board made in laboratory.

#) just_calibration.py: Gets the calibration algorithm by loading a dataset in .pkl. Useful as it does not need to find the points in all frames more than once (done by running point_tracking.py).

#) tabulate_yml_files.py: By reading a list of yml files containing camera calibration parameters, it sorts them and saves it in a .xlsx file.

#) vel_rms.py: Calculates and correlates angular speed with RMS Error from a list targets.

#) parametersCamAIUC.py: File that contains camera calibration parameters for Sony RX0-II owned by AIUC (first camera) with their respective errors.