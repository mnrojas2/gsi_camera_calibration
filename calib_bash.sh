#!/bin/bash

# Original filepath
VIDEO_PATH="/storage/drone/camera_calibration/calibration_tests_20250110/C1_PUC/C0157.MP4"
CALIB_PATH="/storage/drone/camera_calibration/camera_parameters/DSCRX0M2-3178151-20250609-wall.txt"

PTS3D="C0157_3dpoints.txt"
PTS2D="C0157_2dpoints.txt"

# Get basename of the videofile
BASE_NAME=$(basename "$VIDEO_PATH" | cut -f 1 -d '.')

# Run video_tracking script
python -m camera.calibration.video_tracking "$VIDEO_PATH" "$PTS3D" "$PTS2D" "$CALIB_PATH" 

# Extract filename without extension
VIDEO_FOLDER=$(basename "$VIDEO_PATH" .MP4)

# Construct path to the tracked points file
VIDEO_POINTS="$(dirname "$VIDEO_PATH")/${VIDEO_FOLDER}/${VIDEO_FOLDER}_vidpoints.pkl"

# Set the fixed parameters
TS=20

# Run calibration_data script
# Loop from 0 to TS - 1
for ((TF=0; TF<TS; TF++)); do
    echo "Running calibration with -ts $TS and -tf $TF"
    python -m camera.calibration.calibrate_data "$VIDEO_POINTS" -s -ft -ts "$TS" -tf "$TF"
done

# Construct path to the calibration results
CALIB_FOLDER="$(dirname "$VIDEO_PATH")/${VIDEO_FOLDER}/calibrations"

# Run tabulate_calibrations script
python -m camera.calibration.tabulate_calibrations "$CALIB_FOLDER" -e -s

