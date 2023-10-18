import cv2 as cv
import argparse
import glob
import pandas as pd
from matplotlib import pyplot as plt

# Initialize parser
parser = argparse.ArgumentParser(description='Reads yml files in ./results/ and creates a .xlsx with tabulated data.')

# Main
def main():
    args = parser.parse_args()

    calibrationFiles = glob.glob('./tests/results/*.yml')

    cc_data = {}
    cc_summary = {}
    for calibfile in calibrationFiles:
        fs = cv.FileStorage(calibfile, cv.FILE_STORAGE_READ)
        cb = calibfile.split("\\")[-1][:-4]
        mtx = fs.getNode("camera_matrix")
        dcff = fs.getNode("dist_coeff")
        
        summary = fs.getNode("summary")

        fx = mtx.mat()[0,0]
        fy = mtx.mat()[1,1]
        cx = mtx.mat()[0,2]
        cy = mtx.mat()[1,2]

        k1 = dcff.mat().T[0,0]
        k2 = dcff.mat().T[0,1]
        p1 = dcff.mat().T[0,2]
        p2 = dcff.mat().T[0,3]
        k3 = dcff.mat().T[0,4]
        try: # if abs(dcff.mat().T[0,5]) > 0.0:
            k4 = dcff.mat().T[0,4]
            k5 = dcff.mat().T[0,5]
            k6 = dcff.mat().T[0,6]
            cc_data[cb] = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'k1': k1, 'k2': k2, 'p1': p1, 'p2': p2, 'k3': k3, 'k4': k4, 'k5': k5, 'k6': k6}
        except:
            cc_data[cb] = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'k1': k1, 'k2': k2, 'p1': p1, 'p2': p2, 'k3': k3}
        cc_summary[cb] = {'summary': summary.string()}

    df_ccd = pd.DataFrame(cc_data).T
    df_ccd_dcb = df_ccd.describe()
    df_ccd_2 = pd.concat([df_ccd, df_ccd_dcb])
    df_ccd_complete = pd.concat([df_ccd_2, pd.DataFrame(cc_summary).T], axis=1)

    df_ccd_complete.to_excel('./tests/results/camera_calibration_test.xlsx')

    fig, axis = plt.subplots(nrows=2, ncols=2)
    df_ccd['fx'].plot(ax=axis[0,0],kind="hist", title="fx")
    df_ccd['fy'].plot(ax=axis[0,1],kind="hist", title="fy")
    df_ccd['cx'].plot(ax=axis[1,0],kind="hist", title="cx")
    df_ccd['cy'].plot(ax=axis[1,1],kind="hist", title="cy")
    plt.show()

    # with pd.ExcelWriter('results/camera_calibration.xlsx') as writer:  
    #     df_mtx.to_excel(writer, sheet_name='Calibration Matrix')
    #     df_dcff.to_excel(writer, sheet_name='Distortion Coefficients')
    
if __name__=='__main__': main()