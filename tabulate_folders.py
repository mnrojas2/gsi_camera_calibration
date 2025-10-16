#!/usr/bin/env python
import os
import re
import glob
import numpy as np
from matplotlib import pyplot as plt
from tabulate_yml_files import tabulate_data

folder_path = './results/C1_PUC/'
videofolders = [f for f in glob.glob(f'{folder_path}/*') if os.path.isdir(f)]

matrix_dict = {}

for vfolder in videofolders:
    data_coeffs = tabulate_data(vfolder)
    matrix_params_idx = data_coeffs.index[:4].tolist()
    matrix_params = data_coeffs.to_numpy()[:4,:]
    
    dict_aux = {}
    for i in range(len(matrix_params_idx)):
        dict_aux[matrix_params_idx[i]] = matrix_params[i].tolist()

    matrix_dict[os.path.basename(vfolder)] = dict_aux

for idx in matrix_params_idx:
    fig, ax = plt.subplots()
    idx_average = []
    idx_weights = []
    for item in matrix_dict.keys():
        ax.scatter(matrix_dict[item][idx][0], list(matrix_dict.keys()).index(item))
        ax.errorbar(matrix_dict[item][idx][0], list(matrix_dict.keys()).index(item), xerr=matrix_dict[item][idx][1])

        idx_average.append(matrix_dict[item][idx][0])
        idx_weights.append(1/(matrix_dict[item][idx][1])**2)
    
    # Calculate weighted average and variance (1/(sigma**2)) 
    average_idx = np.average(idx_average, weights=idx_weights)
    std_idx = np.sqrt(1/np.sum(idx_weights))

    # Add average as another element of the list
    name_list = list(matrix_dict.keys())
    name_list.append('average')
    ax.vlines(average_idx,0,len(name_list)-1, linestyles='dashed', colors='black')
    ax.scatter(average_idx,len(name_list)-1)
    ax.errorbar(average_idx,len(name_list)-1,xerr=std_idx)

    # Set names to every element in y-axis
    ax.set_yticks(np.arange(len(name_list)))
    ax.set_yticklabels(name_list)

    # Set title
    ax.set_title(f'Parameter: {idx}')
    plt.savefig(f'result_param_{idx}.png', dpi=600)
plt.show()