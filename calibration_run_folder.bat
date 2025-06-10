@echo off
setlocal enabledelayedexpansion

REM Check if the folder argument was provided
if "%~1"=="" (
    echo Usage: calibration_run.bat ^<folder_path^>
    exit /b
)

set "folder=%~1"

REM Loop through each PKL file in the specified folder
for %%F in ("%folder%\*.pkl") do (
    for /L %%i in (0,1,9) do (
        echo Processing %%F with rs: %%i
        python calibration.py %%F -cb .\datasets\2d_coords\DSCRX0M2.txt -ft -rd 20 -rs %%i -s
    )
)

echo All files processed!
pause
