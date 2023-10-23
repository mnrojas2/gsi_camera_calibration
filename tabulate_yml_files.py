#!/usr/bin/env python

import cv2 as cv
import argparse
import glob
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# Initialize parser
parser = argparse.ArgumentParser(description='Reads yml files in ./results/ and creates a .xlsx with tabulated data.')

def gaus(X,C,X_mean,sigma):
    # Calculate the Gaussian PDF values given Gaussian parameters and random variable X
    return C*np.exp(-(X-X_mean)**2/(2*sigma**2))

def df_histogram(dataframe, colname, *args):
    for arg in args:
        idx_filt = arg[0]
        idx_type = arg[1]
    
    # Find rows if a particular str appears in summary or index
    if idx_type == 'index':
        df_idx = dataframe.index.str.contains(idx_filt, na=False)
    elif idx_type == 'summary':
        df_idx = dataframe.summary.str.contains(idx_filt, na=False)
    
    # Filter the original dataframe and calculate mean and std
    new_df = dataframe[df_idx][colname]
    new_df_ds = new_df.describe()
    print(new_df, '\n', new_df_ds)
    
    # Generate histogram
    x_data = new_df.to_numpy(dtype='float32')
    hist, bin_edges = np.histogram(x_data)
    y_hist = hist/sum(hist)

    n = len(y_hist)
    x_hist = np.zeros((n),dtype=float) 
    for ii in range(n):
        x_hist[ii] = (bin_edges[ii+1]+bin_edges[ii])/2

    mean = sum(x_hist*y_hist)/sum(y_hist)                  
    sigma = sum(y_hist*(x_hist-mean)**2)/sum(y_hist) 

    # Gaussian least-square fitting process
    param_optimised,param_covariance_matrix = curve_fit(gaus,x_hist,y_hist,p0=[max(y_hist),mean,sigma],maxfev=5000)

    # Plotting gaussian curve
    fig = plt.figure()
    x_hist_2=np.linspace(np.min(x_hist),np.max(x_hist),500)
    plt.plot(x_hist_2,gaus(x_hist_2,*param_optimised),'r.:',label='Gaussian fit')
    
    weights = np.ones_like(x_data) / len(x_data)
    plt.hist(x_data, weights=weights)
    
    plt.xlabel('Pixels')
    plt.ylabel("Probability")
    plt.title(f"Histogram for '{colname}' in '{idx_filt[:-1]}'")

# Main
def main():
    args = parser.parse_args()

    calibrationFiles = sorted(glob.glob('./results/*.yml'), key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
    
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
    df_ccd_dcb.loc['std/mean'] = df_ccd_dcb.loc['std'] / df_ccd_dcb.loc['mean']
    df_ccd_2 = pd.concat([df_ccd, df_ccd_dcb])
    df_ccd_c = pd.concat([df_ccd_2, pd.DataFrame(cc_summary).T], axis=1)

    # df_ccd_c.to_excel('./results/camera_calibration_test.xlsx')
    
    # print(df_ccd_complete[df_ccd_complete.summary.str.contains("distance,", na=False)])
    df_histogram(df_ccd_c, 'fx', ('C', 'index'))
    df_histogram(df_ccd_c, 'fy', ('C', 'index'))
    df_histogram(df_ccd_c, 'cx', ('C', 'index'))
    df_histogram(df_ccd_c, 'cy', ('C', 'index'))
    plt.show()
    
    df_histogram(df_ccd_c, 'fx', ('Filter by time and points,', 'summary'))
    df_histogram(df_ccd_c, 'fy', ('Filter by time and points,', 'summary'))
    df_histogram(df_ccd_c, 'cx', ('Filter by time and points,', 'summary'))
    df_histogram(df_ccd_c, 'cy', ('Filter by time and points,', 'summary'))
    plt.show()
    
    #Por video, mismo tipo de error, al menos 3 casos
    #Por distancia, mismo video, todos los casos
    #Por tiempo, mismo video, todos los casos
    #Por tiempo y puntos, mismo video, todos los casos.
    
    # fig, axis = plt.subplots(nrows=2, ncols=2)
    # df_ccd['fx'].plot(ax=axis[0,0],kind="hist", title="fx")
    # df_ccd['fy'].plot(ax=axis[0,1],kind="hist", title="fy")
    # df_ccd['cx'].plot(ax=axis[1,0],kind="hist", title="cx")
    # df_ccd['cy'].plot(ax=axis[1,1],kind="hist", title="cy")

    # with pd.ExcelWriter('results/camera_calibration.xlsx') as writer:  
    #     df_mtx.to_excel(writer, sheet_name='Calibration Matrix')
    #     df_dcff.to_excel(writer, sheet_name='Distortion Coefficients')
    
if __name__=='__main__': main()