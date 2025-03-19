#!/usr/bin/env python

import os
import cv2 as cv
import argparse
import glob
import re
import pandas as pd
import numpy as np
import scipy
import datetime
from matplotlib import pyplot as plt
from matplotlib import gridspec


# Initialize parser
parser = argparse.ArgumentParser(description='Reads yml files in a folder and generates histograms and exports data to Excel.')
parser.add_argument('folder', type=str, help='Name of the directory containing the yml files.')
parser.add_argument( '-e', '--excel', action='store_true', default=False, help='Exports data to .xlsx file.')


def df_histogram(dataframe, colname, *args, gauss_c=False, save_values=False):
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
    fig = plt.figure(figsize=(12, 7))
    fig.suptitle('Histograms for all coefficients using all calibration results' if idx_str == ',' else idx_str)
    
    # Generate histograms and add to subplots
    dict_coeff = {}
    for i in range(x_data.shape[1]):
        ax = fig.add_subplot(gs[i])
        weights = np.ones_like(x_data[:,i]) / len(x_data[:,i])
        _, bins, _ = ax.hist(x_data[:,i], weights=weights, density=1, alpha=1)
        
        # _, bins = np.histogram(x_data[:,i])
        # for j in range(6):
        #    ax.hist(x_data[j*28:(j+1)*28,i], bins=bins, density=1, stacked=1, alpha=0.5)
        # gauss_c = False
        
        # Calculate Gaussian least-square fitting process
        if gauss_c:
            mu, sigma = scipy.stats.norm.fit(x_data[:,i])
            best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
            ax.plot(bins, best_fit_line, color='r', label='Gaussian fit')
            
            if save_values:
                dict_coeff[colname[i]] = {'mu': mu, 'sigma': sigma}
        
        if colname[i] not in ['k1', 'k2', 'p1', 'p2', 'k3']:
            ax.set_xlabel('Pixels')
        ax.set_ylabel("Frequency")
        ax.set_title(f'Coefficient {colname[i]}')
    fig.tight_layout()

    if gauss_c and save_values:
        df_coeff = pd.DataFrame.from_dict(dict_coeff, orient='index')
        return df_coeff
    
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
    
    # Get the yml files
    dir = os.path.normpath(args.folder) + '/*.yml'
    calibrationFiles = sorted(glob.glob(dir), key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
    
    # Unpack all data from the files and save them in a dictionary
    cc_data = {}
    cc_summary = {}
    for calibfile in calibrationFiles:
        fs = cv.FileStorage(calibfile, cv.FILE_STORAGE_READ)
        cb = os.path.basename(calibfile)[:-4]
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

    # Generate a dataframe from the dictionary containing all parameters' results 
    df_ccd = pd.DataFrame(cc_data).T
    
    # Get statistics of the dataframe such as mean and std
    df_dcb = df_ccd.describe()
    
    # Add a row for "std/mean" and merge the previous dataframes into one
    df_dcb.loc['std/mean'] = df_dcb.loc['std'] / df_dcb.loc['mean']
    df_ccd_2 = pd.concat([df_ccd, df_dcb])
    df_complete = pd.concat([df_ccd_2, pd.DataFrame(cc_summary).T], axis=1)
    
    # Plot the histogram
    hist_coeffs = df_histogram(df_complete, ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3'], (',', 'summary'), gauss_c=True, save_values=True)
    print(hist_coeffs)
    plt.show()
    
    if args.excel:
        # Save the parameters in an excel file (Excel 2007 - xlsx)
        date_today = str(datetime.datetime.now()).split('.')[0].replace('-', '').replace(':', '').replace(' ', '_')
        
        df_dist = filter_dataframe(df_complete, ('Filter by distance,', '-sm'))
        df_time = filter_dataframe(df_complete, ('Filter by time,', '-sm'))
        df_pnts = filter_dataframe(df_complete, ('Filter by time and points,', '-sm'))
        
        df_dist_full = pd.concat([df_dist, df_dist.describe()])
        df_time_full = pd.concat([df_time, df_time.describe()])
        df_pnts_full = pd.concat([df_pnts, df_pnts.describe()])
        
        with pd.ExcelWriter(f'./results/camera_calibration_{date_today}.xlsx') as writer:
            df_complete.to_excel(writer, sheet_name='Summary')
            df_dist_full.to_excel(writer, sheet_name='Filter by distance')
            df_time_full.to_excel(writer, sheet_name='Filter by time')
            df_pnts_full.to_excel(writer, sheet_name='Filter by time and points')
            
    
if __name__=='__main__': main()