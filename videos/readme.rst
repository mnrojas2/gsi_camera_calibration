========================
Camera Calibration Files
========================


- Files are separated in folders according to the used board.
- There are three type of boards used for calibration:
    
    1) Chessboard: Chessboard of 11x8. 
        
        - Extracted from: https://markhedleyjones.com/projects/calibration-checkerboard-collection
    
    2) Circle grid board: Circlegrid of 11x4. 
    
        - Extracted from: https://longervision.github.io/2017/03/18/ComputerVision/OpenCV/opencv-internal-calibration-circle-grid/
    
    3) New board: Board made in the laboratory.

        - For the located points there are two files (.txt & .csv) containing the same information, as a result from the photogrammetry tests using VSTARS.
        - Order of the first 3 columns are 'X,Y,Z'. The rest are the margin errors for each, not necessary for our current work.
        - The points necessary for calibration are only the ones called TARGETXX and CODEXX (except CODE133 & CODE134).

- Filenames are based in the original filenames of the camera, also indicating the focus length set before shooting them. For the latter, our files are just two types:

    1) Focus set at 0.7 meters (F07): Just for testing purposes to check if the focus really affected the solution in calibration.
    2) Focus set at infinite (Finf): The one used during flights with the drone.

- Other settings used while recording are:

    1) Video Exposure mode: Manual
    2) Shutter speed: 1/1000