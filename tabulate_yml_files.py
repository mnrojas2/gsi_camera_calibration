#!/usr/bin/env python

import cv2 as cv
import argparse
import glob
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.optimize import curve_fit

# Initialize parser
parser = argparse.ArgumentParser(description='Reads yml files in ./results/ and generates histograms and exports data to Excel.')
parser.add_argument( '-e', '--excel', action='store_true', default=False, help='Exports data to .xlsx file.')


def gaus(X,C,X_mean,sigma):
    # Calculate the Gaussian PDF values given Gaussian parameters and random variable X
    return C*np.exp(-(X-X_mean)**2/(2*sigma**2))

def df_histogram(dataframe, colname, *args, gauss_c=False):
    for arg in args:
        idx_str = arg[0]
        idx_type = arg[1]
        
        # Find rows if a particular str appears in summary or index
        if idx_type == 'index':
            df_idx = dataframe.index.str.contains(idx_str, na=False)
        elif idx_type == 'summary':
            df_idx = dataframe.summary.str.contains(idx_str, na=False).to_numpy()

        # If there are more than just one arg, then add them bitwise
        if arg == args[0]:
            df_idx_s = df_idx
        else:
            df_idx_s = df_idx_s & df_idx
    
    # Filter the original dataframe and calculate mean and std
    new_df = dataframe[df_idx_s][colname]
    new_dfs = dataframe[df_idx_s]['summary']
    new_df_ds = new_df.describe()
    new_df_ds.loc['std/mean'] = new_df_ds.loc['std']/new_df_ds.loc['mean']
    print(pd.concat([pd.concat([new_df, new_dfs], axis=1), new_df_ds]).to_string())
    
    # Convert dataframe to numpy array
    x_data = new_df.to_numpy(dtype='float32')
    
    # Define subplots positions
    cols = 3 if x_data.shape[1] >= 5 or x_data.shape[1] == 3 else 2 
    cols = cols if x_data.shape[1] != 1 else 1
    rows = int(np.ceil(x_data.shape[1] / cols))
    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure()
    
    # Generate histograms and add to subplots
    for i in range(x_data.shape[1]):
        hist, bin_edges = np.histogram(x_data[:,i])
        y_hist = hist/sum(hist)
        x_hist = np.zeros((len(y_hist)),dtype=float) 
        
        for ii in range(len(y_hist)):
            x_hist[ii] = (bin_edges[ii+1]+bin_edges[ii])/2

        mean = sum(x_hist*y_hist)/sum(y_hist)                  
        sigma = sum(y_hist*(x_hist-mean)**2)/sum(y_hist) 
        
        ax = fig.add_subplot(gs[i])
        weights = np.ones_like(x_data[:,i]) / len(x_data[:,i])
        ax.hist(x_data[:,i], weights=weights)
        
        # Calculate Gaussian least-square fitting process
        if gauss_c:
            param_optimised, _ = curve_fit(gaus,x_hist,y_hist,p0=[max(y_hist),mean,sigma],maxfev=5000)
            # Plotting gaussian curve
            x_hist_2=np.linspace(np.min(x_hist),np.max(x_hist),500)
            ax.plot(x_hist_2,gaus(x_hist_2,*param_optimised),'r.:',label='Gaussian fit')
        
        ax.set_xlabel('Pixels')
        ax.set_ylabel("Probability")
        ax.set_title(f"'{idx_str[:-1]}' - '{colname[i]}'")
    fig.tight_layout()
    
def filter_dataframe(dataframe, *args):
    for arg in args:
        idx_str = arg[0]
        idx_type = arg[1]
        
        # Find rows if a particular str appears in summary or index
        if idx_type in ['index', '-id']:
            df_idx = dataframe.index.str.contains(idx_str, na=False)
        elif idx_type in ['summary', '-sm']:
            df_idx = dataframe.summary.str.contains(idx_str, na=False).to_numpy()
        
        # If there are more than just one arg, then add them bitwise
        if arg == args[0]:
            df_idx_s = df_idx
        else:
            df_idx_s = df_idx_s & df_idx
    
    # Filter the original dataframe and calculate mean and std
    new_df = dataframe[df_idx_s]
    return new_df
    
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
        nframes = len(fs.getNode("per_view_errors").mat())
        
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
        cc_summary[cb] = {'n.frames': nframes, 'summary': summary.string()}

    df_ccd = pd.DataFrame(cc_data).T
    df_dcb = df_ccd.describe()
    df_dcb.loc['std/mean'] = df_dcb.loc['std'] / df_dcb.loc['mean']
    df_ccd_2 = pd.concat([df_ccd, df_dcb])
    df_complete = pd.concat([df_ccd_2, pd.DataFrame(cc_summary).T], axis=1)
    
    df_histogram(df_complete, ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3'], (',', 'summary'), gauss_c=True)
    plt.show()
    
    if args.excel:
        # df_ccd_c.to_excel('./results/camera_calibration_test.xlsx')
        
        df_dist = filter_dataframe(df_complete, ('Filter by distance,', '-sm'))
        df_time = filter_dataframe(df_complete, ('Filter by time,', '-sm'))
        df_pnts = filter_dataframe(df_complete, ('Filter by time and points,', '-sm'))
        
        with pd.ExcelWriter('results/camera_calibration.xlsx') as writer:
            df_complete.to_excel(writer, sheet_name='Summary')
            df_dist.to_excel(writer, sheet_name='Filter by distance')
            df_time.to_excel(writer, sheet_name='Filter by time')
            df_pnts.to_excel(writer, sheet_name='Filter by time and points')
            
    
if __name__=='__main__': main()