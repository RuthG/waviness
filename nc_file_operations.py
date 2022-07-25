import xarray as xar
import xarray.ufuncs as uf
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, UnivariateSpline
import area_average as aav
from scipy import stats
from scipy.signal import savgol_filter

import pdb

def output_data(dataset, variable_out_list, category_name):
    """Outputs variables in dataset that are contained within variable out list.
    Allows us to save variables that are created within this script, rather than
    saving all variables, including model output."""

    #data_type_folder_name = dataset.data_type

    translate_table = str.maketrans(dict.fromkeys('!@#$/'))

    time_folder_name=str(dataset.start_file)+'_'+str(dataset.end_file)
    directory='/scratch/rg419/derived_data/exps/'+dataset.exp_name.translate(translate_table)+'/'+time_folder_name+'/'+data_type_folder_name+'/' 
                
    if not os.path.exists(directory):
        os.makedirs(directory)

    output_file_name = 'pedram_wave_analysis'+'_'+category_name+'.nc'

    output_dataset = xar.Dataset(coords = dataset.coords)
    
    for key in variable_out_list:
        output_dataset[key] = (dataset[key].dims, dataset[key].values)

    output_dataset.to_netcdf(path=directory+output_file_name)

def load_in_existing_data(dataset, variable_name_list, category_name):
    """Loads in output files from this script."""

    data_type_folder_name = dataset.data_type

    translate_table = str.maketrans(dict.fromkeys('!@#$/'))

    time_folder_name=str(dataset.start_file)+'_'+str(dataset.end_file)
    directory='/scratch/rg419/derived_data/exps/'+dataset.exp_name.translate(translate_table)+'/'+time_folder_name+'/'+data_type_folder_name+'/' 

    input_file_name = 'pedram_wave_analysis'+'_'+category_name+'.nc'

    input_dataset = xar.open_dataset(directory + input_file_name, decode_times=False)
        
    variables_list = [s for s in input_dataset.variables.keys()]

    for key in variables_list:
        if key in variable_name_list:     

            for new_data_dim in input_dataset[key].dims:
                if new_data_dim not in dataset.dims:
                    dataset.coords[new_data_dim] = (new_data_dim, input_dataset[new_data_dim])


            dataset[key] = (input_dataset[key].dims, input_dataset[key].values)

    input_dataset.close()

def read_dataset_isca(base_dir, exp_name, data_type, start_file, end_file, p_level, delta_t=None, opt_substring='', decode_times=True):

    #Want to have options to look at one file for testing
    if start_file==end_file:
            file_name = f'{base_dir}/{exp_name}/run{start_file:04}/plev_{data_type}.nc'
            dataset = xar.open_dataset(file_name, decode_times=decode_times)
            print(f'opened single data file from {file_name}')
    else:
        #Also want an option for multiple input files - either where all the monthly input files have been combined using CDO
        try:
            file_name = f'{base_dir}/{exp_name}/plev_{data_type}_{int(p_level)}_{start_file}_{end_file}{opt_substring}.nc'
            dataset = xar.open_dataset(file_name, decode_times=decode_times)
            print(f'opened single combined data file from {file_name}')
        except:
            #Or if we want to combine the files into one dataset on the fly
            files_to_open = [f'{base_dir}/{exp_name}/run{file_idx:04}/plev_{data_type}_{int(p_level)}{opt_substring}.nc' for file_idx in range(start_file, end_file+1)]
            dataset = xar.open_mfdataset(files_to_open, decode_times=decode_times)
            print(f'opened multi-file dataset starting from {files_to_open[0]}')            


    dataset.attrs['exp_name'] = exp_name
    dataset.attrs['data_type'] = data_type
    dataset.attrs['start_file'] = start_file
    dataset.attrs['end_file'] = end_file    
    dataset.attrs['delta_t'] = delta_t

    return dataset    

def read_dataset(base_dir, exp_name, data_type, start_file, end_file, p_level, filename, delta_t=None):
    """Read JRA-55 data"""

    file_name = f'{base_dir}/{exp_name}/{filename}.nc'
    dataset = xar.open_dataset(file_name)
    print(f'opened single combined data file from {file_name}')
   
    names_dict = {'var2':'slp', 'var33':'ucomp', 'var7':'height', 'var34':'vcomp', 'var11':'temp', 'var39':'omega', 'lev':'pfull',}

    for name in names_dict.keys():
        #new versions of xarray deal with rename differently
        try:          
            if name in dataset.keys():              
                dataset.rename({name:names_dict[name]}, inplace=True)
                if name=='lev':
                    dataset.coords['pfull'] = (('pfull'), dataset.pfull.values/100.)            
        except ValueError:
            if name in dataset.keys():                   
                dataset = dataset.rename({name:names_dict[name]})
                if name=='lev':
                    dataset.coords['pfull'] = dataset.pfull.values/100.

    try:
        dataset = dataset.drop('time_bnds')
    except:
        pass

    dataset.attrs['exp_name'] = exp_name
    dataset.attrs['data_type'] = data_type
    dataset.attrs['start_file'] = start_file
    dataset.attrs['end_file'] = end_file    
    dataset.attrs['delta_t'] = delta_t

    return dataset  

def read_land_file(land_file):

    land_file_ds = xar.open_dataset(land_file)

    return land_file_ds['lsm'].squeeze().values    