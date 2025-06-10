@echo off
setlocal enabledelayedexpansion

REM Check if the folder argument was provided
if "%~1"=="" (
    echo Usage: calibration_run.bat ^<folder_path^>
    exit /b
)

set "file=%~1"

REM Loop through each case of the PKL file
for /L %%i in (0,1,9) do (
    echo Processing %file% with rs: %%i
    python calibration.py %file% -cb .\datasets\2d_coords\DSCRX0M2.txt -ft -rd 20 -rs %%i -s
)

echo All files processed!
pause
