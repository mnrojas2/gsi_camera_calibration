import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2

calibrationFiles = ['./results/calibrationC23F07_upd.yml', './results/calibrationC24F07_upd.yml', './results/calibrationC25F07_upd.yml', 
                    './results/calibrationC26Finf_upd.yml', './results/calibrationC27Finf_upd.yml', './results/calibrationC28Finf_upd.yml',
                    './results/calibrationC32Finf_upd.yml', './results/calibrationC33Finf_upd.yml', './results/calibrationC34Finf_upd.yml',
                    './results/calibrationC29Finf_upd.yml', './results/calibrationC30Finf_upd.yml', './results/calibrationC31Finf_upd.yml'
                    ]

cc_data = {}
for calibfile in calibrationFiles:
    fs = cv2.FileStorage(calibfile, cv2.FILE_STORAGE_READ)
    mtx = fs.getNode("camera_matrix")
    dcff = fs.getNode("dist_coeff")

    fx = mtx.mat()[0,0]
    fy = mtx.mat()[1,1]
    cx = mtx.mat()[0,2]
    cy = mtx.mat()[1,2]

    k1 = dcff.mat()[0,0]
    k2 = dcff.mat()[0,1]
    p1 = dcff.mat()[0,2]
    p2 = dcff.mat()[0,3]
    k3 = dcff.mat()[0,4]
    
    cc_data[calibfile[21:-4]] = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'k1': k1, 'k2': k2, 'p1': p1, 'p2': p2, 'k3': k3}

df_ccd = pd.DataFrame(cc_data).T
df_ccd_dcb = df_ccd.describe()
df_ccd_complete = pd.concat([df_ccd, df_ccd_dcb])

df_ccd_complete.to_excel('results/camera_calibration_3rotcoeff.xlsx')

fig, axis = plt.subplots(nrows=2, ncols=2)
df_ccd['fx'].plot(ax=axis[0,0],kind="hist", title="fx")
df_ccd['fy'].plot(ax=axis[0,1],kind="hist", title="fy")
df_ccd['cx'].plot(ax=axis[1,0],kind="hist", title="cx")
df_ccd['cy'].plot(ax=axis[1,1],kind="hist", title="cy")
plt.show()

# with pd.ExcelWriter('results/camera_calibration_3rotcoeff.xlsx') as writer:  
#     df_mtx.to_excel(writer, sheet_name='Calibration Matrix')
#     df_dcff.to_excel(writer, sheet_name='Distortion Coefficients')