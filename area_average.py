import set_and_get_params as sagp
import numpy as np
import xarray as xar
import pdb

def area_average(dataset, variable_name, model_params, land_ocean_all='all', level=None, axis_in='time', lat_range = None):

    print('performing area average on ',variable_name, 'of type ', land_ocean_all)

    level_to_use = level

    if (variable_name[0:9]=='hc_scaled'):
        variable_name_use=variable_name[10:]
    elif(variable_name[0:8]=='sigma_sb'):
        variable_name_use=variable_name[9:]
    elif('_' in variable_name and '.' in variable_name and variable_name[0:5]=='anom_'):
        split_variable_name=variable_name.split('_')
        variable_name_use = 'anom_'+split_variable_name[1]
        level_to_use = float(split_variable_name[2])
    elif('_' in variable_name and '.' in variable_name):
        split_variable_name=variable_name.split('_')
        variable_name_use = split_variable_name[0]
        level_to_use = float(split_variable_name[1])        
    else:
        variable_name_use=variable_name

    if(level_to_use!=None):
        data_input=dataset[variable_name_use].sel(pfull=level_to_use, method='nearest').load()
        lev_string = '_'+str(int(np.around(level_to_use*100.))/100.)
    else:
        data_input=dataset[variable_name_use]
        lev_string = ''


    if (variable_name[0:9]=='hc_scaled'):
        data_to_average=data_input*dataset['ml_heat_cap']/model_params['delta_t']
    elif(variable_name[0:8]=='sigma_sb'):
        data_to_average=model_params['sigma_sb']*data_input**4.
    else:
        data_to_average=data_input

    try:
        grid_area=dataset['grid_cell_area']
    except KeyError:
        sagp.get_grid_sizes(dataset,model_params)
        grid_area=dataset['grid_cell_area']

    land_ocean_all_str_suffix = land_ocean_all

    if(land_ocean_all == 'land'):
        scaled_grid_area=grid_area*(dataset['land'])

        multiplied=scaled_grid_area*data_to_average
        average=multiplied.sum(('lat','lon'))/scaled_grid_area.sum(('lat','lon'))

    elif(land_ocean_all == 'ocean'):
        scaled_grid_area=grid_area*(1.-dataset['land'])

        multiplied=scaled_grid_area*data_to_average
        average=multiplied.sum(('lat','lon'))/scaled_grid_area.sum(('lat','lon'))

    elif(land_ocean_all == 'ocean_non_ice'):
        scaled_grid_area=grid_area*(1.-dataset['land_ice_mask'])

        multiplied=scaled_grid_area*data_to_average
        average=multiplied.sum(('lat','lon'))/scaled_grid_area.sum(('lat','lon'))

    elif(land_ocean_all == 'all'):
        multiplied=grid_area*data_to_average
        average=multiplied.sum(('lat','lon'))/grid_area.sum(('lat','lon'))

    elif(land_ocean_all == 'qflux_area'):
        scaled_grid_area=grid_area*(dataset['qflux_area'])

        multiplied=scaled_grid_area*data_to_average
        average=multiplied.sum(('lat','lon'))/scaled_grid_area.sum(('lat','lon'))

    elif(land_ocean_all[3:] == 'eur'):
        scaled_grid_area=grid_area*(dataset[land_ocean_all])

        multiplied=scaled_grid_area*data_to_average
        average=multiplied.sum(('lat','lon'))/scaled_grid_area.sum(('lat','lon'))

    elif(land_ocean_all == 'land_lat_range'):
        scaled_grid_area = grid_area*(dataset['land']).where((dataset.lat > lat_range[0]) & (dataset.lat < lat_range[1]))
        
        multiplied = scaled_grid_area * data_to_average
        average=multiplied.sum(('lat','lon'))/scaled_grid_area.sum(('lat','lon'))     
        land_ocean_all_str_suffix = land_ocean_all_str_suffix + '_' + str(lat_range[0])+'_' + str(lat_range[1])

    elif(land_ocean_all == 'lat_range'):
        scaled_grid_area = grid_area.where((dataset.lat > lat_range[0]) & (dataset.lat < lat_range[1]))
        
        multiplied = scaled_grid_area * data_to_average
        average=multiplied.sum(('lat','lon'))/scaled_grid_area.sum(('lat','lon'))     
        land_ocean_all_str_suffix = land_ocean_all_str_suffix + '_' + str(lat_range[0])+'_' + str(lat_range[1])

    elif(land_ocean_all == 'just_lat_range'):
        scaled_grid_area = grid_area.where((dataset.lat > lat_range[0]) & (dataset.lat < lat_range[1]))
        
        multiplied = scaled_grid_area * data_to_average
        average=multiplied.sum(('lat'))/scaled_grid_area.sum(('lat'))     
        land_ocean_all_str_suffix = land_ocean_all_str_suffix + '_' + str(lat_range[0])+'_' + str(lat_range[1])

    elif(land_ocean_all[0:18]=='specified_area_av_'):
        scaled_grid_area=grid_area*(dataset[land_ocean_all])

        multiplied=scaled_grid_area*data_to_average
        average=multiplied.sum(('lat','lon'), skipna=False)/scaled_grid_area.sum(('lat','lon')) #N.b. that default skipna is sort of true, meaning a sum of all nans will return 0, rather than nan. Would rather it returned nan for clarity, so putting skipna = False.
    else:
        print('invalid area-average option: ',land_ocean_all)
        return

    if level != level_to_use :
        new_var_name=variable_name_use+lev_string+'_area_av_'+land_ocean_all_str_suffix
    else:
        new_var_name=variable_name+lev_string+'_area_av_'+land_ocean_all_str_suffix

    try:
        if len(average.dims) ==1:
            dataset[new_var_name]=((axis_in), average)
        else:    
            dataset[new_var_name]=((data_to_average.dims[0:2]), average)
    except:
        try:
            dims_ordered = []
            if 'time' in average.dims:
                dims_ordered.append('time')
            if 'pfull' in average.dims:
                dims_ordered.append('pfull')            
            if 'lat' in average.dims:
                dims_ordered.append('lat')            
            if 'lon' in average.dims:
                dims_ordered.append('lon')              
            dataset[new_var_name]=(tuple(dims_ordered), average.transpose(*dims_ordered))        
        except:
            dataset[new_var_name]=(average.dims, average)        

    
def european_area_av(dataset, model_params, eur_area_av_input):

    variables_list=eur_area_av_input['variables_list']
    try:
            levels_list  = eur_area_av_input['levels_list']
    except KeyError:
        levels_list  = None

    lats=dataset.lat
    lons=dataset.lon

    lon_array, lat_array = np.meshgrid(lons,lats)

    idx_nw_eur =     (45. <= lat_array) & (lat_array < 60.) & (-5. < lon_array) & (lon_array < 27.5)
    idx_nw_eur_neg = (45. <= lat_array) & (lat_array < 60.) & (np.mod(-5.,360.) < lon_array)

    idx_sw_eur =     (30. <= lat_array) & (lat_array < 45.) & (-5. < lon_array) & (lon_array < 27.5)
    idx_sw_eur_neg = (30. <= lat_array) & (lat_array < 45.) & (np.mod(-5.,360.) < lon_array)

    idx_ne_eur = (45. <= lat_array) & (lat_array < 60.) & (27.5 < lon_array) & (lon_array < 60.)
    idx_se_eur = (30. <= lat_array) & (lat_array < 45.) & (27.5 < lon_array) & (lon_array < 60.)

    idx_all_eur =     (35. <= lat_array) & (lat_array < 60.) & (-10. < lon_array) & (lon_array < 40.)
    idx_all_eur_neg = (35. <= lat_array) & (lat_array < 60.) & (np.mod(-10.,360.) < lon_array)


    land_nw_eur=np.zeros_like(dataset.land)
    land_sw_eur=np.zeros_like(dataset.land)

    land_ne_eur=np.zeros_like(dataset.land)
    land_se_eur=np.zeros_like(dataset.land)
    
    land_all_eur=np.zeros_like(dataset.land)
    
    land_nw_eur[idx_nw_eur]=1.0
    land_nw_eur[idx_nw_eur_neg]=1.0

    land_sw_eur[idx_sw_eur]=1.0
    land_sw_eur[idx_sw_eur_neg]=1.0

    land_ne_eur[idx_ne_eur]=1.0
    land_se_eur[idx_se_eur]=1.0

    land_all_eur[idx_all_eur]=1.0
    land_all_eur[idx_all_eur_neg]=1.0

    dataset['nw_eur']=(('lat','lon'), land_nw_eur)
    dataset['sw_eur']=(('lat','lon'), land_sw_eur)
    dataset['ne_eur']=(('lat','lon'), land_ne_eur)
    dataset['se_eur']=(('lat','lon'), land_se_eur)

    dataset['al_eur']=(('lat','lon'), land_all_eur)
    

    for i in range(np.shape(variables_list)[0]):
        var_name=variables_list[i]
        if levels_list!=None:
            level_in=levels_list[i]
        else:
            level_in=None

        area_average(dataset, var_name, model_params, land_ocean_all='nw_eur',level=level_in)
        area_average(dataset, var_name, model_params, land_ocean_all='sw_eur',level=level_in)
        area_average(dataset, var_name, model_params, land_ocean_all='ne_eur',level=level_in)
        area_average(dataset, var_name, model_params, land_ocean_all='se_eur',level=level_in)
        area_average(dataset, var_name, model_params, land_ocean_all='al_eur',level=level_in)


def specified_area_av(dataset, model_params, lat_range, lon_range, other_inputs, carry_out_averaging = True, rename_area=None):

    variables_list=other_inputs['variables_list']
    try:
            levels_list  = other_inputs['levels_list']
    except KeyError:
        levels_list  = None

    try:
        only_over_land = other_inputs['only_over_land']
        only_over_land_str = '_over_land'
    except KeyError:
        only_over_land = False
        only_over_land_str = '_over_land_and_ocean'


    lats=dataset.lat
    lons=dataset.lon

    lon_array, lat_array = np.meshgrid(lons,lats)
    land_array = dataset['land'].values

    if only_over_land:
        idx_nw_eur =     (lat_range[0] <= lat_array) & (lat_array < lat_range[1]) & (lon_range[0] < lon_array) & (lon_array < lon_range[1]) & (land_array == 1.)
    else:
        idx_nw_eur =     (lat_range[0] <= lat_array) & (lat_array < lat_range[1]) & (lon_range[0] < lon_array) & (lon_array < lon_range[1])        

    if lon_range[0] < 0.:
        if only_over_land:        
            idx_nw_eur_neg = (lat_range[0] <= lat_array) & (lat_array < lat_range[1]) & (np.mod(lon_range[0],360.) < lon_array) & (land_array == 1.)
        else:
            idx_nw_eur_neg = (lat_range[0] <= lat_array) & (lat_array < lat_range[1]) & (np.mod(lon_range[0],360.) < lon_array)            
    else:
        idx_nw_eur_neg = []

    land_nw_eur=np.zeros_like(dataset.land)
    
    land_nw_eur[idx_nw_eur]=1.0
    land_nw_eur[idx_nw_eur_neg]=1.0

    var_name_area =f'specified_area_av_{lat_range[0]}_{lat_range[1]}_{lon_range[0]}_{lon_range[1]}'+only_over_land_str

    dataset[var_name_area]=(('lat','lon'), land_nw_eur)
   
    if carry_out_averaging:
        for i in range(np.shape(variables_list)[0]):
            var_name=variables_list[i]
            if levels_list!=None:
                level_in=levels_list[i]
            else:
                level_in=None

            area_average(dataset, var_name, model_params, land_ocean_all=var_name_area,level=level_in)
    else:
        if rename_area is not None:
            dataset[rename_area]=(('lat','lon'), land_nw_eur)


    return '_area_av_'+var_name_area

def qflux_area_av(dataset, model_params, qflux_area_av_input):

    qflux_area=np.zeros_like(dataset.land)

    variables_list     = qflux_area_av_input['variables_list']

    warmpool_lat_centre= qflux_area_av_input['lat_centre']
    warmpool_lon_centre= qflux_area_av_input['lon_centre']

    warmpool_width     = qflux_area_av_input['width']
    warmpool_width_lon = qflux_area_av_input['width_lon']

    lats=dataset.lat
    lons=dataset.lon

    latbs=dataset.latb
    lonbs=dataset.lonb


    for j in np.arange(len(lats)):
         lat = 0.5*(latbs[j+1] + latbs[j])
         lat = (lat-warmpool_lat_centre)/warmpool_width
         for i in np.arange(len(lons)):
              lon = 0.5*(lonbs[i+1] + lonbs[i])
              lon = (lon-warmpool_lon_centre)/warmpool_width_lon
              if( lat**2.+lon**2. <= 1.0 ):
                  qflux_area[j,i]=1.0

    dataset['qflux_area']=(('lat','lon'), qflux_area)

    for i in range(np.shape(variables_list)[0]):
        var_name=variables_list[i]
        original_axes_of_data = dataset[var_name].dims
        list_of_dims = list(original_axes_of_data)
        try:
            list_of_dims.remove('lat')        
            list_of_dims.remove('lon')            
        except:
            print('original data doesnt have lat lon as dimension - problem!')
        tuple_of_dims = tuple(list_of_dims)
        
        area_average(dataset, var_name, model_params, land_ocean_all='qflux_area', axis_in = tuple_of_dims)

def vertical_integral(dataset_in, var_name, model_params, vertical_average=False):

    var_to_integrate = dataset_in[var_name]

    # take a diff of half levels, and assign to pfull coordinates
    dp=xar.DataArray(dataset_in.phalf.diff('phalf').values*100, coords=[('pfull', dataset_in.pfull)])
    product = var_to_integrate*dp
    product.load()
    vert_sum=np.sum(product, axis=product.dims.index('pfull'))
    
    dims_in = var_to_integrate.dims
    dims_in_list = list(dims_in)
    dims_in_list.remove('pfull')
    dims_out = tuple(dims_in_list)
    
    dataset_in[var_name+'_vert_int']=(dims_out, vert_sum)       

    if vertical_average:
        dataset_in[var_name+'_vert_av'] = (dims_out, vert_sum / np.sum(dp))

def vertical_integral_log_p_ps(dataset_in, var_name, model_params):

    var_to_integrate = dataset_in[var_name]

    # take a diff of half levels, and assign to pfull coordinates

    sigma = dataset_in['phalf']/dataset_in['phalf'].sel(phalf=np.max(dataset_in.phalf.values)).values

    log_sigma = np.log(sigma)

    diff_log_sigma = log_sigma.diff('phalf')

    not_valid = np.isfinite(diff_log_sigma) !=True

    if np.any(not_valid):
        diff_log_sigma[np.where(not_valid)] = 0.0

    dp=xar.DataArray(diff_log_sigma.values, coords=[('pfull', dataset_in.pfull)])
    product = var_to_integrate*dp
    product.load()
    vert_sum=np.sum(product, axis=product.dims.index('pfull'))
    
    dims_in = var_to_integrate.dims
    dims_in_list = list(dims_in)
    dims_in_list.remove('pfull')
    dims_out = tuple(dims_in_list)
    
    dataset_in[var_name+'_vert_int_d_log_p_ps']=(dims_out, vert_sum)       

def global_area_and_press_average(dataset_in, var_name, model_params):

    area_average(dataset_in, var_name, model_params)
    vertical_integral(dataset_in, var_name+'_area_av_all', model_params, vertical_average=True)

def total_water_content(dataset_in, model_params, grav_in=None):

    if grav_in is None:
        grav_in = model_params['grav']
    
    epsilon = model_params['rdgas']/model_params['rvgas']
    
    constants_out_front = -(1./grav_in)
    
    q = dataset_in['sphum'].load()
    
    quantity_top = q*(epsilon + q*(1-epsilon))
    quantity_bottom = q+ epsilon*(1-q)
    
    quantity = -1.*constants_out_front * quantity_top/quantity_bottom #-1 at front is because dp in vertical integral will be positive, which is like integrating from top down. But I want bottom up, so needs to be minus. 
    
    dataset_in['water_content'] = (quantity.dims, quantity)
    
    global_area_and_press_average(dataset_in, 'water_content', model_params)

def total_water_content_mk2(dataset_in, model_params, grav_in=None):

    if grav_in is None:
        grav_in = model_params['grav']
    
    epsilon = model_params['rdgas']/model_params['rvgas']
    
    constants_out_front = (1./grav_in)

    area_average(dataset_in, 'sphum', model_params)

    q_area_av = dataset_in['sphum_area_av_all'].load()

    e = (q_area_av * dataset_in.pfull*100.)/(q_area_av + epsilon*(1.-q_area_av))

    dataset_in['e_water_vapour_content_calc'] = (e.dims, e)

    vertical_integral_log_p_ps(dataset_in, 'e_water_vapour_content_calc', model_params)

    quantity = constants_out_front * epsilon* dataset_in['e_water_vapour_content_calc_vert_int_d_log_p_ps']

    
    dataset_in['water_content_mk2'] = (quantity.dims, quantity)
    
