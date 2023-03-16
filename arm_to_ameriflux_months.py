# Imports
import os
import json
import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt
import glob
import act
import dask

site = ['sgp']
fac = ['E39']
datastreams = ['ecorsf', 'sebs', 'amc', 'stamp', 'stamppcp', ]
year = ['202201', '202202', '202203']
sdate = '20220101'
edate = '20221231'

# Creating a def to handle missing values if the data doesn't exist. 
def missing_var(time):
    """
    Returns an array of -9999 that's the same shape as time
    """
    var_array = np.ones((time.shape))*-9999
    return var_array


# Sets the dictionary for the current ameriflux names for ARM sites
ameriflux_names = {
    'sgpC1': 'US-A14',
    'sgpE33': 'US-A33',
    'sgpE37': 'US-A37',
    'sgpE39': 'US-A39',
    'sgpE41': 'US-A41',
    'nsaE10': 'US-A10'
}

# Define variable mapping and units
# This maps from ARM names to AmeriFlux names using
# https://ameriflux.lbl.gov/data/aboutdata/data-variables/
var_mapping = {
    'co2_flux': {'name': 'FC', 'units': 'umol m-2 s-1'},
    'co2_molar_fraction': {'name': 'CO2', 'units': 'nmol mol-1'},
    'co2_mixing_ratio': {'name': 'CO2_MIXING_RATIO', 'units': 'umol mol-1'},
    'h2o_mole_fraction': {'name': 'H2O', 'units': 'mmol mol-1'},
    'h2o_mixing_ratio': {'name': 'H2O_MIXING_RATIO', 'units': 'mmol mol-1'},
    'ch4_mole_fraction': {'name': 'CH4', 'units': 'nmol mol-1'},
    'ch4_mixing_ratio': {'name': 'CH4_MIXING_RATIO','units': 'nmol mol-1'},
    'momentum_flux': {'name': 'TAU', 'units': 'kg m-1 s-2'},
    'sensible_heat_flux': {'name': 'H', 'units': 'W m-2'},
    'latent_heat_flux': {'name': 'LE', 'units': 'W m-2'},
    'air_temperature': {'name': 'TA', 'units': 'deg C'},
    'air_pressure': {'name': 'PA', 'units': 'kPa'},
    'relative_humidity': {'name': 'RH', 'units': '%'},
    'sonic_temperature': {'name': 'T_SONIC', 'units': 'deg C'},
    'water_vapor_pressure_defecit': {'name': 'VPD', 'units': 'hPa'},
    'Monin_Obukhov_length': {'name': 'MO_LENGTH', 'units': 'm'},
    'Monin_Obukhov_stability_parameter': {'name': 'ZL', 'units': ''},
    'mean_wind': {'name': 'WS', 'units': 'm s-1'},
    'wind_direction_from_north': {'name': 'WD', 'units': 'deg'},
    'friction_velocity': {'name': 'USTAR', 'units': 'm s-1'},
    'maximum_instantaneous_wind_speed': {'name': 'WS_MAX', 'units': 'm s-1'},
    'down_short_hemisp': {'name': 'SW_IN', 'units': 'W m-2'},
    'up_short_hemisp': {'name': 'SW_OUT', 'units': 'W m-2'},
    'down_long': {'name': 'LW_IN', 'units': 'W m-2'},
    'up_long': {'name': 'LW_OUT', 'units': 'W m-2'},
    'albedo': {'name': 'ALB', 'units': '%'},
    'net_radiation': {'name': 'NETRAD', 'units': 'W m-2'},
    'par_inc': {'name': 'PPFD_IN', 'units': 'umol m-2 s-1'},
    'par_ref': {'name': 'PPFD_OUT', 'units': 'umol m-2 s-1'},
    'precip': {'name': 'P', 'units': 'mm'},
}

# The soil mapping is tricky as there are multiple positional qualifiers to map to
# Section 3.3 explains this more: https://ameriflux.lbl.gov/data/aboutdata/data-variables/
# This just maps the base variable names to AmeriFlux.  Sections below map it out more.
# It needs to be flexible as things like the AMC won't be at all sites
soil_mapping = {
    'surface_soil_heat_flux': {'name': 'G', 'units': 'W m-2'},
    'vwc': {'name': 'SWC', 'units': '%'},
    'soil_temp': {'name': 'TS', 'units': 'deg C'},
    'temp': {'name': 'TS', 'units': 'deg C'},
}

# Loop through each site/facility to create the data product
for s in site:
    for f in fac:
        # Sets the ameriflux name and data path
        af_name = ameriflux_names[s+f] 
        data_path = ('./data/' + s +'ameriflux' + f + '/')
        # Empty list to add in xarray datasets for merging
        ds_all = []
        # Empty dictionary for the data to convert to pandas dataframe later on
        data = {}
        # Process data for each datastream
        for dst in datastreams:
            print('Reading in ', dst)
            files = []

            # NOTE - Not implemented here but with the time shift needing to occur for the ECOR
            # to algin time_bounds across the variables, the last record will be missing ECOR data
            # We need to pull in the next file to fill that last time stamp in
            for y in year:
                files += glob.glob('/data/archive/' + s + '/' + s + dst + f + '.b1/*' + y + '*')
            files.sort()

            # Read in data to an xarray object for easy merging
            ds = act.io.armfiles.read_netcdf(files, parallel=True)

            # Add DQR information to object to remove later
            #ds = act.qc.arm.add_dqr_to_qc(ds)

            # Filter all the failing QC out
            # IMPORTANT!!!!!  While not implemented here yet, the flag_[variable_name] variables will
            # also need to be used to further QA/QC the data
            ds.qcfilter.datafilter(del_qc_var=False, rm_assessments=['Bad', 'Incorrect', 'Indeterminate', 'Suspect'])

            # Adjust the timestamp to start at the beginning of the sampling period
            # Mainly impacts the ECOR data
            if 'time_bounds' in ds:
                ds = act.utils.datetime_utils.adjust_timestamp(ds)
                ds = ds.rename({'time_bounds': 'time_bounds_' + dst})

            # Resample the data to 30 minutes
            ds = ds.resample(time='30Min').mean()

            # Update all variable units to that defined by AmeriFlux
            for v in ds:
                if v in var_mapping:
                    ds = ds.utils.change_units(variables=v, desired_unit=var_mapping[v]['units'])

            # Add to object for merging datasets at the end
            ds_all.append(ds)

        # Merge 4 instruments together into one xarray object
        ds_merge = xr.merge(ds_all, compat='override')

        # Get times and format to AmeriFlux standards
        ts_start = ds_merge['time'].dt.strftime('%Y%m%d%H%M').values
        ts_end = [pd.to_datetime(t +  np.timedelta64(30, 'm')).strftime('%Y%m%d%H%M') for t in ds_merge['time'].values]

        # Add to the data dictionary as the first 2 columns
        data['TIMESTAMP_START'] = ts_start
        data['TIMESTAMP_END'] = ts_end

        # Run through the variable mapping to add the name/data to the dictionary
        # If not available, add -9999
        for v in var_mapping:
            if v in ds_merge:
                data[var_mapping[v]['name']] = ds_merge[v].values
            else:
                data[var_mapping[v]['name']] = missing_var(ds_merge['time'].values)

        # Perform the soil variable mapping
        prev_var = ''
        for var in soil_mapping:
            # For each new variable the vertical should be 1
            vert = 1
            # If the AmeriFlux variable type changes, reset qualifiers (i.e. VWC to TS)
            if soil_mapping[var]['name'] != prev_var:
                h = 1
                r = 1
                prev_var = soil_mapping[var]['name']
            # Find corresponding soil variables in the merged dataset
            soil_vars = [v2 for v2 in list(ds_merge) if (v2.startswith(var)) & ('std' not in v2) & ('qc' not in v2) & ('net' not in v2)]

            # Run through each variable and add to the dataset as appropriate
            for i, svar in enumerate(soil_vars):
                if ('avg' in svar) | ('average' in svar):
                    continue
                soil_data = ds_merge[svar].values
                data_shape = soil_data.shape
                # If data has depth like the STAMP, there is a need to update the vertical
                # qualifier and product new names
                if len(data_shape) > 1:
                    vert = 1
                    coords = ds_merge[svar].coords
                    depth_name = list(coords)[-1]
                    depth_values = ds_merge[depth_name].values
                    for depth_ind in range(len(depth_values)):
                        soil_data_depth = soil_data[:, depth_ind]
                        new_name = '_'.join([soil_mapping[var]['name'], str(h), str(vert), str(r)])
                        data[new_name] = soil_data_depth
                        vert += 1
                else:
                    new_name = '_'.join([soil_mapping[var]['name'], str(h), str(vert), str(r)])
                    data[new_name] = soil_data
                h += 1

        # Create a dataframe of the data and fill in NaN values
        df = pd.DataFrame(data)
        df = df.fillna(-9999.)

        # Remove the first row as it will be for the previous day with the ECOR time shift
        df = df.iloc[1:, :]

        # Write data out
        directory = './data/' + s + 'mergedflux' + f +'/'
        outfile = af_name + '_HH' + str(ts_start[0]) + '_' + str(ts_end[0]) + '.csv'
        if not os.path.exists(directory):
            os.makedirs(directory)
        df.to_csv(directory + outfile, index=False)
