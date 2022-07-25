import xarray as xr
import EquivLatitude as eql
import numpy as np
from scipy.interpolate import CubicSpline, interp2d
import gauss_grid as gg
from tqdm import tqdm
import os

def calculate_local_wave_activity(dataset, quantity_2d_arr, monotonic_decreasing=True, planet_radius=6371.e3, n_quantity_levels=40, LWA_area=False):
    """Calculates the local wave activity following the formalism described in 10.1002/2015GL066959. 
        Main difference from standard local wave activity defined in 10.1175/JAS-D-15-0194.1 is that this is 
        designed to take Z500 as input. On this basis, we would set monotonic decreasing =True, reflecting the fact that
        Z500 decreases as it goes poleward. This is necessary as the equivalent latitude code was designed for montonic
        increasing quantities like PV."""

    lat_arr = dataset['lat']
    lon_arr = dataset['lon']

    lon_arr_2d, lat_arr_2d = np.meshgrid(lon_arr, lat_arr)

    ntime = quantity_2d_arr.time.shape[0]
    nlat  = lat_arr.shape[0]
    nlon  = lon_arr.shape[0]    

    print('running equiv lat calculation')

    if monotonic_decreasing:
        #Supply equivalent latitude calculation with -quantity as -quantity is monotonic increasing towards the pole
        #Main advantage of equivalent latitude calculation is it gives a monotonic function of quantity as a function of equivalent latitude
        eq_lat, zlev = eql.eqlat(-quantity_2d_arr.values, lat_arr.values, lon_arr.values, n_quantity_levels)
    else:
        #Supply equivalent latitude calculation with quantity as quantity is monotonic increasing towards the pole        
        eq_lat, zlev = eql.eqlat(quantity_2d_arr.values, lat_arr.values, lon_arr.values, n_quantity_levels)

    print('finished equiv lat calculation')

    #Problem with eq_lat is that it's on a grid defined by the number of quantity levels we asked for (n_quantity_levels)
    #So we will now interpolate zlev onto the normal latitude grid lat_arr using the correspondence of eq_lat and zlev
    #to train the liner interpolation 

    eq_lat_arr = np.asarray(eq_lat)
    zlev_arr = np.asarray(zlev)
    z_mean_state = np.zeros((ntime, nlat))

    faw_south = np.zeros((ntime, nlat, nlon))
    faw_north = np.zeros((ntime, nlat, nlon))    

    for t_tick in range(ntime):
        #sometimes equivalent lat calculation can produce both nan entries and repeated entries, neither of which work in the interpolation
        #to counter this we first subset for unique equivalent latitude entries
        unique_eq_lats, where_unique = np.unique(eq_lat_arr[t_tick,:], return_index=True)
        z_lev_unique = zlev_arr[t_tick, where_unique]
        #then we substitute for finite ones
        where_finite = np.where(np.isfinite(unique_eq_lats))[0]
        cs_obj = CubicSpline(unique_eq_lats[where_finite], z_lev_unique[where_finite])
        #ask cubic spline to give z values on native lat grid.
        if monotonic_decreasing:
            #Have to reverse sign to give quantity back, rather than -quantity
            z_mean_state[t_tick,:] = -cs_obj(lat_arr.values)
        else:
            z_mean_state[t_tick,:] = cs_obj(lat_arr.values)

    print('finished interpolating to native grid')

    #This is the mean state defined by the equivalent latitude calculation
    dataset['height_500_mean_state'] = (('time', 'lat'), z_mean_state)

    print('finished anom calculation')

    #loop over 
    for lat_tick in tqdm(range(nlat)):

        #To calculate lwa-south we first apply a where for latitude being less than latitude defined by lat_tick
        #then we subtract off the value of the mean state at this latitude
        #Finally, we then only select values of the anomaly that are negative
        anom_arr_south_temp = (quantity_2d_arr).where(dataset['lat']<=lat_arr[lat_tick]) - dataset['height_500_mean_state'][:,lat_tick]
        anom_arr_south = anom_arr_south_temp.where(anom_arr_south_temp<=0.)

        #we now do the same for lwa-north, with greater thans instead of less thans
        anom_arr_north_temp = (quantity_2d_arr).where(dataset['lat']>=lat_arr[lat_tick])- dataset['height_500_mean_state'][:,lat_tick]
        anom_arr_north = anom_arr_north_temp.where(anom_arr_north_temp>=0.)

        #In order to do the integral, we multiply by cos(lat) with lat defined by lat-tick, and delta-lat in radians
        if LWA_area:
            scaled_anom_arr_south = (anom_arr_south*0. +1.) * np.cos(np.deg2rad(lat_arr)) * np.deg2rad(dataset['delta_lat_arr'])
            scaled_anom_arr_north = (anom_arr_north*0. +1.) * np.cos(np.deg2rad(lat_arr)) * np.deg2rad(dataset['delta_lat_arr'])                
        else:
            scaled_anom_arr_south = anom_arr_south * np.cos(np.deg2rad(lat_arr)) * np.deg2rad(dataset['delta_lat_arr'])
            scaled_anom_arr_north = anom_arr_north * np.cos(np.deg2rad(lat_arr)) * np.deg2rad(dataset['delta_lat_arr'])                
        
        
        #we then perform the latitude integral for each lat-tick by summing over lat
        #N.B. we're using the latitudes here as we've regridded the equivalent latitudes onto the original grid.
        #Get rid of values poleward of 85 so that 1/cos(lat)
        #doesn't give crazy answers.
        if np.abs(lat_arr[lat_tick])<=85.:
            faw_south[:, lat_tick, :] = (planet_radius/np.cos(np.deg2rad(lat_arr[lat_tick]))) *(scaled_anom_arr_south.sum('lat'))
            faw_north[:, lat_tick, :] = (planet_radius/np.cos(np.deg2rad(lat_arr[lat_tick]))) *(scaled_anom_arr_north.sum('lat'))    
        else: 
            faw_south[:, lat_tick, :] = np.nan
            faw_north[:, lat_tick, :] = np.nan            

    dataset['faw_south'] = (('time', 'lat', 'lon'), faw_south)
    dataset['faw_north'] = (('time', 'lat', 'lon'), faw_north)
    
    #define local wave activity as south - north
    dataset['lwa'] = (dataset['faw_south'].dims, (dataset['faw_south'].values - dataset['faw_north'].values) )

    #define sum so that we get information on phase, as in 10.1002/2015GL066959 figure 2
    dataset['lwa_sum'] = (dataset['faw_south'].dims, (dataset['faw_south'].values + dataset['faw_north'].values) )




def setup_grid(dataset, gaussian_grid):
    '''If a gaussian grid is requested define a latitude array with this spacing. 
       Otherwise assume constant and subtract lats to find spacing'''
    if gaussian_grid:
        latitudes, latitude_bounds  = gg.gaussian_latitudes(int(dataset['lat'].values.shape[0]/2))
        delta_lat = [dl[1]-dl[0] for dl in latitude_bounds]
        dataset['delta_lat_arr'] = ('lat', delta_lat)
    else:
        delta_lat = np.abs(dataset['lat'][1].values-dataset['lat'][0].values)
        dataset['delta_lat_arr'] = ('lat', np.zeros_like(dataset['lat'].values)+ delta_lat)
        


def run_local_wave_activity_calc(file_name, outdir_name, field_name='height', p_level=500., gaussian_grid=True, decode_times=False, 
                                 latdim='lat', londim='lon', pdim='pfull', geopotential=False, LWA_area=False):
    
    dataset = xr.open_dataset(file_name, decode_times=decode_times)   # Load up data
    dataset = dataset.rename({latdim:'lat', londim:'lon', pdim:'pfull'})
    dataset = dataset.sortby('lat') # Make sure latitude goes from negative to positive
    
    setup_grid(dataset, gaussian_grid)
    
    #import time
    #applying this latitude mask is very slow - consider subsetting to nh only using cdo or equivalent before calculating lwa
    if dataset['lat'].min()<0.:
        #start = time.process_time()
        lats = [dataset.lat[i].values for i in range(len(dataset.lat)) if dataset.lat[i] >= 0 and dataset.lat[i] <= 90]    
        dataset = dataset.sel(lat=lats)
        #print(time.process_time() - start)
        #start = time.process_time()
        #dataset=dataset.where(dataset.lat>=0., drop=True)
        #print(time.process_time() - start)

    print('applied lat mask')

    print('finished loading data')

    category_name = f'local_wave_activity_z{int(p_level)}'
    variable_name_list = ['lwa', 'lwa_sum', 'faw_north', 'faw_south']

    print('Calculating LWA')       
    
    quantity_2d_arr = dataset[field_name].sel(pfull=p_level).squeeze()

    if geopotential:
        quantity_2d_arr = quantity_2d_arr/9.81

    calculate_local_wave_activity(dataset, quantity_2d_arr, LWA_area=LWA_area)

    #saves the output data so that it can be easily loaded again
    directory = '/scratch/rg419/derived_data/exps/' + outdir_name + '/daily/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if LWA_area:
        output_file_name = 'LWA_area_sit.nc'
    else:
        output_file_name = 'LWA_sit.nc'
    
    output_dataset = xr.Dataset(coords = dataset.coords)
    
    for key in variable_name_list:
        output_dataset[key] = (dataset[key].dims, dataset[key].values)

    output_dataset.to_netcdf(path=directory+output_file_name)

    return dataset



def groupsequence(lst):
    '''Python3 program to Find groups of strictly increasing numbers within from https://www.geeksforgeeks.org/python-find-groups-of-strictly-increasing-numbers-in-a-list/'''
    
    res = [[lst[0]]]
  
    for i in range(1, len(lst)):
        if lst[i-1]+1 == lst[i]:
            res[-1].append(lst[i])
  
        else:
            res.append([lst[i]])
    return res




if __name__=="__main__":

    #file_name = '/disco/share/rg419/ERA_5/processed/geopotential_1979_2020_daymean_500.nc'
    #dataset = run_local_wave_activity_calc(file_name, 'ERA_5', field_name='z', p_level=500., gaussian_grid=False, 
    #                                       latdim='latitude', londim='longitude', pdim='level', geopotential=True, LWA_area=True)

    file_name = '/disco/share/rg419/ERA_5/processed/geopotential_1979_2020_daymean_500.nc'
    dataset = run_local_wave_activity_calc(file_name, 'ERA_5', field_name='z', p_level=500., gaussian_grid=False, 
                                           latdim='latitude', londim='longitude', pdim='level', geopotential=True)
    
    


