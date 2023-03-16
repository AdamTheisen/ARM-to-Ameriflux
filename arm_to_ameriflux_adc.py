#!/usr/bin/env python
"""
 NAME:
   arm_to_ameriflux.py
 PURPOSE:
   To pull data from the ARM data stream for ECORSF, SEBS, AMC, STAMP, and
   STAMPCP data and merge data into one file. From that the data is then 
   renamed to meet the Ameriflux naming convention and data units are 
   converted. Then site identification is renamed to meet Ameriflux standards.
 SYNTAX:
   python arm_to_ameriflux.py
"""
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
sdate = '20220101'
edate = '20221231'

# Creating a def to handle missing values if the data doesn't exist. 
def missing_var(var_name, time):
    """
    This function creates an array of 1's that is the same length as the 
    TIMESTAMP_END variable, which in this current case is 48, then multiplies 
    it by -9999 for a missing value code. The array is converted into an xarray 
    data array with it's dimensions being time.The missing value array is 
    renamed to match the variable name given by Ameriflux. 
    """
    var_array = np.ones((time.shape))*-9999
    return var_name


# Set dates, first one is to use for downloading data, second is ARM format
# This section can be changed to read in current data with datetime to 
# eliminate manual date selection.
ameriflux_names = {
    'sgpC1': 'US-A14',
    'sgpE33': 'US-A33',
    'sgpE37': 'US-A37',
    'sgpE39': 'US-A39',
    'sgpE41': 'US-A41',
    'nsaE10': 'US-A10'
}

# Define variable mapping and units
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
time_vars = ['TIMESTAMP_START', 'TIMESTAMP_END']
soil_mapping = {
    'surface_soil_heat_flux': {'name': 'G', 'units': 'W m-2'},
    'vwc': {'name': 'SWC', 'units': '%'},
    'soil_temp': {'name': 'TS', 'units': 'deg C'},
    'temp': {'name': 'TS', 'units': 'deg C'},
    #'soil_temperature_west': {'name': 'TS', 'units': 'deg C'},
    #'soil_temperature_east': {'name': 'TS', 'units': 'deg C'},
    #'soil_temperature_south': {'name': 'TS', 'units': 'deg C'},
}
c1 = [var_mapping[v]['name'] for v in var_mapping]
columns = time_vars + c1
dates = act.utils.dates_between(sdate, edate)
dates.sort()
for s in site:
    for f in fac:
        af_name = ameriflux_names[s+f] 
        data_path = ('./data/' + s +'ameriflux' + f + '/')
        for d in dates:
            ds_all = []
            data = {}
            for dst in datastreams:
                fdate = d.strftime('%Y%m%d')
                files = glob.glob('/data/archive/' + s + '/' + s + dst + f + '.b1/*' + fdate + '*')
                ds = act.io.armfiles.read_netcdf(files, parallel=True)
                #ds = act.qc.arm.add_dqr_to_qc(ds)
                ds.qcfilter.datafilter(del_qc_var=False, rm_assessments=['Bad', 'Incorrect', 'Indeterminate', 'Suspect'])
                if 'time_bounds' in ds:
                    ds = act.utils.datetime_utils.adjust_timestamp(ds)
                    ds = ds.rename({'time_bounds': 'time_bounds_' + dst})
                ds = ds.resample(time='30Min').mean()
                for v in ds:
                    if v in var_mapping:
                        ds = ds.utils.change_units(variables=v, desired_unit=var_mapping[v]['units'])

                # Add to object for merging datasets at the end
                ds_all.append(ds)

            # Merge 4 instruments together into one xarray object
            ds_merge = xr.merge(ds_all, compat='override')
            ts_start = ds_merge['time'].dt.strftime('%Y%m%d%H%M').values
            ts_end = [pd.to_datetime(t +  np.timedelta64(30, 'm')).strftime('%Y%m%d%H%M') for t in ds_merge['time'].values]
            data['TIMESTAMP_START'] = ts_start
            data['TIMESTAMP_END'] = ts_end
            for v in var_mapping:
                if v in ds_merge:
                    data[var_mapping[v]['name']] = ds_merge[v].values
                else:
                    data[var_mapping[v]['name']] = missing_var(var_mapping[v]['name'], ds_merge['time'].values)

            prev_var = ''
            for var in soil_mapping:
                vert = 1
                if soil_mapping[var]['name'] != prev_var:
                    h = 1
                    r = 1
                    prev_var = soil_mapping[var]['name']
                soil_vars = [v2 for v2 in list(ds_merge) if (v2.startswith(var)) & ('std' not in v2) & ('qc' not in v2) & ('net' not in v2)]
                for i, svar in enumerate(soil_vars):
                    if ('avg' in svar) | ('average' in svar):
                        continue
                    soil_data = ds_merge[svar].values
                    data_shape = soil_data.shape
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

            df = pd.DataFrame(data)
            df = df.fillna(-9999.)
            df = df.iloc[1:, :]
            directory = './data/' + s + 'mergedflux' + f +'/'
            outfile = af_name + '_HH' + str(ts_start[0]) + '_' + str(ts_end[0]) + '.csv'
            if not os.path.exists(directory):
                os.makedirs(directory)
            df.to_csv(directory + outfile, index=False)
