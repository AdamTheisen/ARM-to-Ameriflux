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
import numpy as np
import json
import xarray as xr
import glob
import act

# Creating a def to handle missing values if the data doesn't exist. 
def missing_var(var_name):
    """
    This function creates an array of 1's that is the same length as the 
    TIMESTAMP_END variable, which in this current case is 48, then multiplies 
    it by -9999 for a missing value code. The array is converted into an xarray 
    data array with it's dimensions being time.The missing value array is 
    renamed to match the variable name given by Ameriflux. 
    """
    var_array = np.ones((TIMESTAMP_END.shape))*-9999
    var_xr_array = xr.DataArray(var_array, dims='time')
    var_name = var_xr_array.rename(var_name)
    return var_name


# Read in ARM Live Data Webservice Token and Username
with open('./token.json') as f:
    data = json.load(f)
username = data['username']
token = data['token']

# Set the facility and instruments to use
facs = ['E39']
site_ID = str(facs[0])
site_name = 'sgp'

data_type = '.b1'

# Instrument being used.
datastreams = ['ecorsf', 'sebs', 'amc', 'stamp', 'stamppcp', ]

# Set dates, first one is to use for downloading data, second is ARM format
# This section can be changed to read in current data with datetime to 
# eliminate manual date selection.
date = '2020-07-10'
sdate = ''.join(date.split('-'))
for f in facs:
    ecor = None
    sebs = None
    amc = None
    obj = []

    # Look for ECOR data, download if None, and read in
    ecor_files = glob.glob('./'+site_name+'ecorsf' +
                           f + data_type + '/*' + sdate + '*')
    if len(ecor_files) == 0:
        ecor_files = act.discovery.download_data(
            username, token, site_name + 'ecorsf' + f + data_type, date, date)
    if len(ecor_files) > 0:
        ecor = act.io.armfiles.read_netcdf(ecor_files)
        ecor = ecor.rename({'time_bounds': 'time_bounds_ecor'})
        # Add to object for merging datasets at the end
        obj.append(ecor)

    # Look for SEBS data, download if None, and read in
    sebs_files = glob.glob('./'+site_name+'sebs' + f +
                           data_type+'/*' + sdate + '*')
    if len(sebs_files) == 0:
        sebs_files = act.discovery.download_data(
            username, token, site_name+'sebs' + f + data_type, date, date)
    if len(sebs_files) > 0:
        sebs = act.io.armfiles.read_netcdf(sebs_files)
        obj.append(sebs)

    # Look for AMC data, download if None, and read in
    amc_files = glob.glob('./'+site_name+'amc' + f +
                          data_type+'/*' + sdate + '*')
    if len(amc_files) == 0:
        amc_files = act.discovery.download_data(
            username, token, site_name+'amc' + f + data_type, date, date)
    if len(amc_files) > 0:
        amc = act.io.armfiles.read_netcdf(amc_files)
        amc = amc.rename({'time_bounds': 'time_bounds_amc'})
        obj.append(amc)

    # Look for STAMP data, download if None, and read in
    stamp_files = glob.glob('./'+site_name+'stamp' +
                            f + data_type+'/*' + sdate + '*')
    if len(stamp_files) == 0:
        stamp_files = act.discovery.download_data(
            username, token, site_name+'stamp' + f + data_type, date, date)
    if len(stamp_files) > 0:
        stamp = act.io.armfiles.read_netcdf(stamp_files)
        obj.append(stamp)

    # Look for STAMPPCP data, download if None, and read in
    stamppcp_files = glob.glob(
        './'+site_name+'stamppcp' + f + data_type+'/*' + sdate + '*')
    if len(stamppcp_files) == 0:
        stamppcp_files = act.discovery.download_data(
            username, token, site_name+'stamppcp' + f + data_type, date, date)
    if len(stamppcp_files) > 0:
        stamppcp = act.io.armfiles.read_netcdf(stamppcp_files)
        stamppcp = stamppcp.resample(time='30T').mean()
        obj.append(stamppcp)

    # Merge 4 instruments together into one xarray object
    obj = xr.merge(obj, compat='override')

    # Close out individual objects
    ecor.close()
    sebs.close()
    amc.close()
    stamp.close()
    stamppcp.close()

    # Create dirs and write out merged data
    if not os.path.exists('./'+site_name+'flux' + f + data_type+'/'):
        os.mkdir('./'+site_name+'flux' + f + data_type+'/')
    obj.to_netcdf('./'+site_name+'flux' + f + data_type+'/' +
                  site_name+'flux' + f + data_type + sdate + '.000000.nc')


# Assigning the correct Ameriflux Site name based on location.
# Central Facility, Lamont, OK
if (site_name == 'sgp' and site_ID == 'C1'):
    Ameriflux_name = 'US-A14'
# Newkirk, OK (Extended)
elif (site_name == 'sgp' and site_ID == 'E33'):
    Ameriflux_name = 'US-A33'
# Waukomis, OK (Extended)
elif (site_name == 'sgp' and site_ID == 'E37'):
    Ameriflux_name = 'US-A37'
# Morrison, OK (Extended)
elif (site_name == 'sgp' and site_ID == 'E39'):
    Ameriflux_name = 'US-A39'
# Peckham, OK (Extended)
elif (site_name == 'sgp' and site_ID == 'E41'):
    Ameriflux_name = 'US-A41'

# Barrow, AK
elif site_name == 'nsa' and site_ID == 'C1':
    Ameriflux_name = 'US-A10'

# Path to data in the local computer. This will need to be modified to match
# the local computer's path
FLUX_path = (r'Z:/Matt/virtual_machine_shared/Ameriflux/' +
             site_name+'fluxE39'+data_type+'/')
FLUX_files = os.listdir(FLUX_path)

# Looping through the sorted files in the FLUX_files path to open and read
# the data. 
for files in sorted(FLUX_files):
    fpath_FLUX = os.path.join(FLUX_path, files)
    ds = xr.open_dataset(fpath_FLUX, engine='netcdf4')

    # Setting up time bounds and created an array of formatted times to
    # meet Ameriflux standard.
    time_bounds = ds['time_bounds_ecor'].dt.strftime("%Y%m%d%H%M")
    time_start = time_bounds
    TIMESTAMP_START = np.array([])
    TIMESTAMP_END = np.array([])
    for i in range(len(time_bounds)):
        start = time_bounds[i][0]
        end = time_bounds[i][1]
        TIMESTAMP_START = np.append(TIMESTAMP_START, start)
        TIMESTAMP_END = np.append(TIMESTAMP_END, end)

    # Adding variable and renaming it in xarray
    TIMESTAMP_START = xr.DataArray(TIMESTAMP_START, dims='time')
    TIMESTAMP_START = TIMESTAMP_START.rename("TIMESTAMP_START")
    TIMESTAMP_END = xr.DataArray(TIMESTAMP_END, dims='time')
    TIMESTAMP_END = TIMESTAMP_END.rename("TIMESTAMP_END")


# -----ECORSF Data. Ameriflux data variable assigned for consistency.------

    # Carbon Dioxide (CO2) turbulent flux (no storage correction)
    try:
        FC = ds['co2_flux'].rename("FC")
    except KeyError:
        FC = missing_var("FC")

    # Methane (CH4) turbulent flux (no storage correction)
    # FCH4 Not in our data

    # Carbon Dioxide (CO2) mole fraction in wet air
    try:
        CO2 = ds['co2_molar_fraction'].rename("CO2")
    except KeyError:
        CO2 = missing_var("CO2")


    # Carbon Dioxide (CO2) in mole fraction of dry air
    try:
        CO2_MIXING_RATIO = ds['co2_mixing_ratio'].rename("CO2_MIXING_RATIO")

    except KeyError:
        CO2_MIXING_RATIO = missing_var("CO2_MIXING_RATIO")

    # Water (H2O) vapor in mole fraction of wet air
    try:
        H2O = ds['h2o_molar_fraction'].rename("H20")
    except KeyError:
        H2O = missing_var("H2O")
        
    # Water (H2O) vapor in mole fraction of dry air
    try:
        H2O_MIXING_RATIO = ds['h2o_mixing_ratio'].rename("H2O_MIXING_RATIO")
    except KeyError:
        H2O_MIXING_RATIO = missing_var("H2O_MIXING_RATIO")

    # Methane (CH4) mole fraction in wet air
    try:
        CH4 = ds['ch4_molar_fraction'].rename("CH4")
    except KeyError:
        CH4 = missing_var("CH4")

    # Methane (CH4) in mole fraction of dry air
    try:
        CH4_MIXING_RATIO = ds['ch4_mixing_ratio'].rename("CH4_MIXING_RATIO")
    except KeyError:
        CH4_MIXING_RATIO = missing_var("CH4_MIXING_RATIO")

    # Momentum flux
    try:
        TAU = ds['momentum_flux'].rename("TAU")
    except KeyError:
        TAU = missing_var("TAU")
        
    # Sensible heat turbulent flux (no storage correction)
    try:
        H = ds['sensible_heat_flux'].rename("H")
    except KeyError:
        H = missing_var("H")

    # Latent heat turbulent flux (no storage correction)
    try:
        LE = ds['latent_flux'].rename("LE")
    except KeyError:
        LE = missing_var("LE")

    # Air temperature given in K then converted to degC
    try:
        TA = ds['air_temperature'].rename("TA")
        TA = TA-273.15
    except KeyError:
        TA = missing_var("TA")

    # Atmospheric pressure
    try:
        PA = ds['air_pressure'].rename("PA")
    except KeyError:
        PA = missing_var("PA")

    # Relative humidity, range 0-100
    try:
        RH = ds['relative_humidity'].rename("RH")
    except KeyError:
        RH = missing_var("RH")

    # Sonic temperature
    try:
        T_SONIC = ds['sonic_temperature'].rename("T_SONIC")
        T_SONIC = T_SONIC-273.15
    except KeyError:
        T_SONIC = missing_var("T_SONIC")

    # Vapor Pressure Deficit and is converted to hPa from kPa
    try:
        VPD = ds['water_vapor_pressure_deficit'].rename("VPD")
        VPD = VPD*10
    except KeyError:
        VPD = missing_var("VPD")

    # Monin-Obukhov length
    try:
        MO_LENGTH = ds['Monin_Obukhov_length'].rename("MO_LENGTH")
    except KeyError:
        MO_LENGTH = missing_var('MO_LENGTH')

    # Monin-Obukhov Stability parameter
    try:
        ZL = ds['Monin_Obukhov_stability_parameter'].rename("ZL")
    except KeyError:
        ZL = missing_var("ZL")
        
    # Wind speed
    try:
        WS = ds['mean_wind'].rename("WS")
    except KeyError:
        WS = missing_var("WS")

    # Wind direction
    try:
        WD = ds['wind_direction_from_north'].rename("WD")
    except KeyError:
        WD = missing_var("WD")
        
    # Friction velocity
    try:
        USTAR = ds['friction_velocity'].rename("USTAR")
    except KeyError:
        USTAR = missing_var("USTAR")

    # Maximum WS in the averaging period
    try:
        WS_MAX = ds['maximum_instantaneous_wind_speed'].rename("WS_MAX")
    except KeyError:
        WS_MAX = missing_var("WS_MAX")

# -----SEBS Data. Ameriflux data variable assigned for consistency.------
    # Shortwave radiation, incoming
    try:
        SW_IN = ds['down_short_hemisp'].rename("SW_IN")
    except KeyError:
        SW_IN = missing_var("SW_IN")

    # Shortwave radiation, outgoing
    try:
        SW_OUT = ds['up_short_hemisp'].rename("SW_OUT")
    except KeyError:
        SW_OUT = missing_var("SW_OUT")

    # Longwave radiation, incoming
    try:
        LW_IN = ds['down_long'].rename("LW_IN")
    except KeyError:
        LW_IN = missing_var("LW_IN")

    # Longwave radiation, outgoing
    try:
        LW_OUT = ds['up_long'].rename("LW_OUT")
    except KeyError:
        LW_OUT = missing_var("LW_OUT")

    # Albedo, range 0-100
    try:
        ALB = ds['albedo'].rename("ALB")
    except KeyError:
        ALB = missing_var("ALB")
    # Net radiation
    try:
        NETRAD = ds['net_radiation'].rename("NETRAD")
    except KeyError:
        NETRAD = missing_var("NETRAD")

    # Soil heat flux
    try:
        G_1_1_1 = ds['surface_soil_heat_flux_1'].rename("G_1_1_1")
    except KeyError:
        G_1_1_1 = missing_var("G_1_1_1")

    # Soil heat flux
    try:
        G_1_1_2 = ds['surface_soil_heat_flux_2'].rename("G_1_1_2")
    except KeyError:
        G_1_1_2 = missing_var("G_1_1_2")
        
    # Soil heat flux
    try:
        G_1_1_3 = ds['surface_soil_heat_flux_3'].rename("G_1_1_3")
    except KeyError:
        G_1_1_3 = missing_var("G_1_1_3")

    # Soil heat flux average. Variable is tested to ensure no missing value
    # codes skew the averaging.Two data points are pulled to make sure data
    # exists in the file.
    count = 0
    if G_1_1_1[0] != -9999 and G_1_1_1[30] != -9999:
        G_1_1_1 = G_1_1_1
        count = count+1
    else:
        G_1_1_1_1 = G_1_1_1*0

    if G_1_1_2[0] != -9999 and G_1_1_2[30] != -9999:
        G_1_1_2 = G_1_1_2
        count = count+1
    else:
        G_1_1_2 = G_1_1_2*0

    if G_1_1_3[0] != -9999 and G_1_1_3[30] != -9999:
        G_1_1_3 = G_1_1_3
        count = count+1
    else:
        G_1_1_3 = G_1_1_3*0

    G_1_1_A = (G_1_1_1 + G_1_1_2 + G_1_1_3)/count
    G_1_1_A = G_1_1_A.rename("G_1_1_A")

    # Soil temperature
    try:
        TS_1_1_1 = ds['soil_temp_1'].rename("TS_1_1_1")
    except KeyError:
        TS_1_1_1 = missing_var("TS_1_1_1")
        
    # Soil temperature
    try:
        TS_1_1_2 = ds['soil_temp_2'].rename("TS_1_1_2")
    except KeyError:
        TS_1_1_2 = missing_var("TS_1_1_2")
        
    # Soil temperature
    try:
        TS_1_1_3 = ds['soil_temp_3'].rename("TS_1_1_3")
    except KeyError:
        TS_1_1_3 = missing_var("TS_1_1_3")

    # Soil temperature average. Variables are tested to verify they exist
    # and ensure no missing value codes skew the data.
    count = 0
    if TS_1_1_1[0] != -9999 and TS_1_1_1[30] != -9999:
        TS_1_1_1 = TS_1_1_1
        count = count+1
    else:
        TS_1_1_1 = TS_1_1_1*0

    if TS_1_1_2[0] != -9999 and TS_1_1_2[30] != -9999:
        TS_1_1_2 = TS_1_1_2
        count = count+1
    else:
        TS_1_1_2 = TS_1_1_2*0

    if TS_1_1_3[0] != -9999 and TS_1_1_3[30] != -9999:
        TS_1_1_3 = TS_1_1_3
        count = count+1
    else:
        TS_1_1_3 = TS_1_1_3*0

    # Averages out the soil temperature based on number of non missing
    # value data given in the array.
    TS_1_1_A = (TS_1_1_1 + TS_1_1_2 + TS_1_1_3)/count
    TS_1_1_A = TS_1_1_A.rename("TS_1_1_A")


# -----AMC Data. Ameriflux data variable assigned for consistency.------

    # Soil Temperature from AMC probe depth -36.8 cm.
    try:
        TS_2_2_1 = ds['temp_1'].rename("TS_2_2_1")
    except KeyError:
        TS_2_2_1 = missing_var("TS_2_2_1")

    # Soil temperature from AMC probe depth -14 cm.
    try:
        TS_2_1_1 = ds['temp_2'].rename("TS_2_1_1")
    except KeyError:
        TS_2_1_1 = missing_var("TS_2_1_1")

    # Soil temperature from AMC probe depth -35.6 cm
    try:
        TS_2_2_2 = ds['temp_3'].rename("TS_2_2_2")
    except KeyError:
        TS_2_2_2 = missing_var("TS_2_2_2")

    # Soil temperature from AMC probe depth -14 cm
    try:
        TS_2_1_2 = ds['temp_4'].rename("TS_2_1_2")
    except KeyError:
        TS_2_1_2 = missing_var("TS_2_1_2")

    # Soil temperature from AMC probe depth -35.6 cm
    try:
        TS_2_2_3 = ds['temp_5'].rename("TS_2_2_3")
    except KeyError:
        TS_2_2_3 = missing_var("TS_2_2_3")

    # Soil temperature from AMC probe depth -14 cm
    try:
        TS_2_1_3 = ds['temp_6'].rename("TS_2_1_3")
    except KeyError:
        TS_2_1_3 = missing_var("TS_2_1_3")

    # Soil temperature from AMC probe depth -34.3 cm
    try:
        TS_2_2_4 = ds['temp_7'].rename("TS_2_2_4")
    except KeyError:
        TS_2_2_4 = missing_var("TS_2_2_4")

    # Soil temperature from AMC probe depth -15.2 cm
    try:
        TS_2_1_4 = ds['temp_8'].rename("TS_2_1_4")
    except KeyError:
        TS_2_1_4 = missing_var("TS_2_1_4")

    # Soil temperature from AMC probe depth -34.9
    try:
        TS_2_2_5 = ds['temp_9'].rename("TS_2_2_5")
    except KeyError:
        TS_2_2_5 = missing_var("TS_2_2_5")

    # Soil temperature from AMC probe depth -14 cm
    try:
        TS_2_1_5 = ds['temp_10'].rename("TS_2_1_5")
    except KeyError:
        TS_2_1_5 = missing_var("TS_2_1_5")

    # Soil temperature from AMC probe depth -34.3 cm
    try:
        TS_2_2_6 = ds['temp_11'].rename("TS_2_2_6")
    except KeyError:
        TS_2_2_6 = missing_var("TS_2_2_6")

    # Soil temperature from AMC probe depth -16.5 cm
    try:
        TS_2_1_6 = ds['temp_12'].rename("TS_2_1_6")
    except KeyError:
        TS_2_1_6 = missing_var("TS_2_1_6")

    # Temperature averaging for probes (-14 to -16 cm) below ground surface.
    # The code checks to verify the data exists and for the averaging. If the
    # data is -9999, the value is multipled by zero.
    count = 0
    if TS_2_1_1[0] != -9999 and TS_2_1_1[30] != -9999:
        TS_2_1_1 = TS_2_1_1
        count = count+1
    else:
        TS_2_1_1 = TS_2_1_1*0

    if TS_2_1_2[0] != -9999 and TS_2_1_2[30] != -9999:
        TS_2_1_2 = TS_2_1_2
        count = count+1
    else:
        TS_2_1_2 = TS_2_1_2*0

    if TS_2_1_3[0] != -9999 and TS_2_1_3[30] != -9999:
        TS_2_1_3 = TS_2_1_3
        count = count+1
    else:
        TS_2_1_3 = TS_2_1_3*0

    if TS_2_1_4[0] != -9999 and TS_2_1_4[30] != -9999:
        TS_2_1_4 = TS_2_1_4
        count = count+1
    else:
        TS_2_1_4 = TS_2_1_4*0

    if TS_2_1_5[0] != -9999 and TS_2_1_5[30] != -9999:
        TS_2_1_5 = TS_2_1_5
        count = count+1
    else:
        TS_2_1_5 = TS_2_1_5*0

    if TS_2_1_6[0] != -9999 and TS_2_1_6[30] != -9999:
        TS_2_1_6 = TS_2_1_6
        count = count+1
    else:
        TS_2_1_6 = TS_2_1_6*0

    # The averaging is done here and then renames the variable in xarray.
    TS_2_1_A = (TS_2_1_1 + TS_2_1_2 + TS_2_1_3 + TS_2_1_4 +
                TS_2_1_5 + TS_2_1_6)/count
    TS_2_1_A = TS_2_1_A.rename("TS_2_1_A")

    # Temperature averaging for probes below -30 cm. Same averaging method
    # is applied here as previous steps.
    count = 0
    if TS_2_2_1[0] != -9999 and TS_2_2_1[30] != -9999:
        TS_2_2_1 = TS_2_2_1
        count = count+1
    else:
        TS_2_2_1 = TS_2_2_1*0

    if TS_2_2_2[0] != -9999 and TS_2_2_2[30] != -9999:
        TS_2_2_2 = TS_2_2_2
        count = count+1
    else:
        TS_3_2 = TS_2_2_2*0

    if TS_2_2_3[0] != -9999 and TS_2_2_3[30] != -9999:
        TS_2_2_3 = TS_2_2_3
        count = count+1
    else:
        TS_2_2_3 = TS_2_2_3*0

    if TS_2_2_4[0] != -9999 and TS_2_2_4[30] != -9999:
        TS_2_2_4 = TS_2_2_4
        count = count+1
    else:
        TS_2_2_4 = TS_2_2_4*0

    if TS_2_2_5[0] != -9999 and TS_2_2_5[30] != -9999:
        TS_2_2_5 = TS_2_2_5
        count = count+1
    else:
        TS_2_2_5 = TS_2_2_5*0

    if TS_2_2_6[0] != -9999 and TS_2_2_6[30] != -9999:
        TS_2_2_6 = TS_2_2_6
        count = count+1
    else:
        TS_2_2_6 = TS_2_2_6*0

    # Averaging and renaming the variable.
    TS_2_2_A = (TS_2_2_1 + TS_2_2_2 + TS_2_2_3 + TS_2_2_4 + TS_2_2_5 +
                TS_2_2_6)/count
    TS_2_2_A = TS_2_2_A.rename("TS_2_2_A")

    # Photosynthetic photon flux density, incoming
    try:
        PPFD_IN = ds['par_inc'].rename("PPFD_IN")
    except KeyError:
        PPFD_IN = missing_var("PPFD_IN")

    # Photosynthetic photon flux density, outgoing
    try:
        PPFD_OUT = ds['par_ref'].rename("PPFD_OUT")
    except KeyError:
        PPFD_OUT = missing_var("PPFD_OUT")

    # Soil water content (volumetric), range 0-100
    # Soil water content at -36.8 cm
    try:
        SWC_1_2_1 = ds['vwc_1'].rename("SWC_1_2_1")
    except KeyError:
        SWC_1_2_1 = missing_var("SWC_1_2_1")

    # Soil water content at -14 cm
    try:
        SWC_1_1_1 = ds['vwc_2'].rename("SWC_1_1_1")
    except KeyError:
        SWC_1_1_1 = missing_var("SWC_1_1_1")
        
    # Soil water content at -35.6 cm
    try:
        SWC_1_2_2 = ds['vwc_3'].rename("SWC_1_2_2")
    except KeyError:
        SWC_1_2_2 = missing_var("SWC_1_2_2")

    # Soil water content at -14 cm
    try:
        SWC_1_1_2 = ds['vwc_4'].rename("SWC_1_1_2")
    except KeyError:
        SWC_1_1_2 = missing_var("SWC_1_1_2")

    # Soil water content at -35.6 cm
    try:
        SWC_1_2_3 = ds['vwc_5'].rename("SWC_1_2_3")
    except KeyError:
        SWC_1_2_3 = missing_var("SWC_1_2_3")

    # Soil water content at -14 cm
    try:
        SWC_1_1_3 = ds['vwc_6'].rename("SWC_1_1_3")
    except KeyError:
        SWC_1_1_3 = missing_var("SWC_1_1_3")

    # Soil water content at -34.3 cm
    try:
        SWC_1_2_4 = ds['vwc_7'].rename("SWC_1_2_4")
    except KeyError:
        SWC_1_2_4 = missing_var("SWC_1_2_4")

    # Soil water content at -15.2 cm
    try:
        SWC_1_1_4 = ds['vwc_8'].rename("SWC_1_1_4")
    except KeyError:
        SWC_1_1_4 = missing_var("SWC_1_1_4")

    # Soil water content at -34.9 cm
    try:
        SWC_1_2_5 = ds['vwc_9'].rename("SWC_1_2_5")
    except KeyError:
        SWC_1_2_5 = missing_var("SWC_1_2_5")

    # Soil water content at -14 cm
    try:
        SWC_1_1_5 = ds['vwc_10'].rename("SWC_1_1_5")
    except KeyError:
        SWC_1_1_5 = missing_var("SWC_1_1_5")

    # Soil water content at -34.3 cm
    try:
        SWC_1_2_6 = ds['vwc_11'].rename("SWC_1_2_6")
    except KeyError:
        SWC_1_2_6 = missing_var("SWC_1_2_6")

    # Soil water content at -16.5 cm
    try:
        SWC_1_1_6 = ds['vwc_12'].rename("SWC_1_1_6")
    except KeyError:
        SWC_1_1_6 = missing_var("SWC_1_1_6")

    # Soil water content averaging for probes closer (-14 to -16 cm)
    # to ground surface.
    count = 0
    if SWC_1_1_1[0] != -9999 and SWC_1_1_1[30] != -9999:
        SWC_1_1_1 = SWC_1_1_1
        count = count+1
    else:
        SWC_1_1 = SWC_1_1_1*0

    if SWC_1_1_2[0] != -9999 and SWC_1_1_2[30] != -9999:
        SWC_1_1_2 = SWC_1_1_2
        count = count+1
    else:
        SWC_1_1_2 = SWC_1_1_2*0

    if SWC_1_1_3[0] != -9999 and SWC_1_1_3[30] != -9999:
        SWC_1_1_3 = SWC_1_1_3
        count = count+1
    else:
        SWC_1_1_3 = SWC_1_1_3*0

    if SWC_1_1_4[0] != -9999 and SWC_1_1_4[30] != -9999:
        SWC_1_1_4 = SWC_1_1_4
        count = count+1
    else:
        SWC_1_1_4 = SWC_1_1_4*0

    if SWC_1_1_5[0] != -9999 and SWC_1_1_5[30] != -9999:
        SWC_1_1_5 = SWC_1_1_5
        count = count+1
    else:
        SWC_1_1_5 = SWC_1_1_5*0

    if SWC_1_1_6[0] != -9999 and SWC_1_1_6[30] != -9999:
        SWC_1_1_6 = SWC_1_1_6
        count = count+1
    else:
        SWC_1_1_6 = SWC_1_1_6*0

    # Averaging equation for soil water content.
    SWC_1_1_A = (SWC_1_1_1 + SWC_1_1_2 + SWC_1_1_3 +
                 SWC_1_1_4 + SWC_1_1_5 + SWC_1_1_6)/count
    SWC_1_1_A = SWC_1_1_A.rename("SWC_1_1_A")

    # Soil water content averaging for probes between -34.0 cm and -36.0 cm.
    count = 0
    if SWC_1_2_1[0] != -9999 and SWC_1_2_1[30] != -9999:
        SWC_1_2_1 = SWC_1_2_1
        count = count+1
    else:
        SWC_1_2_1 = SWC_1_2_1*0

    if SWC_1_2_2[0] != -9999 and SWC_1_2_2[30] != -9999:
        SWC_1_2_2 = SWC_1_2_2
        count = count+1
    else:
        SWC_1_2_2 = SWC_1_2_2*0

    if SWC_1_2_3[0] != -9999 and SWC_1_2_3[30] != -9999:
        SWC_1_2_3 = SWC_1_2_3
        count = count+1
    else:
        SWC_1_2_3 = SWC_1_2_3*0

    if SWC_1_2_4[0] != -9999 and SWC_1_2_4[30] != -9999:
        SWC_1_2_4 = SWC_1_2_4
        count = count+1
    else:
        SWC_1_2_4 = SWC_1_2_4*0

    if SWC_1_2_5[0] != -9999 and SWC_1_2_5[30] != -9999:
        SWC_1_2_5 = SWC_1_2_5
        count = count+1
    else:
        SWC_1_2_5 = SWC_1_2_5*0

    if SWC_1_2_6[0] != -9999 and SWC_1_2_6[30] != -9999:
        SWC_1_2_6 = SWC_1_2_6
        count = count+1
    else:
        SWC_1_2_6 = SWC_1_2_6*0

    # Averaging equation for soil water content
    SWC_1_2_A = (SWC_1_2_1 + SWC_1_2_2 + SWC_1_2_3 +
                 SWC_1_2_4 + SWC_1_2_5 + SWC_1_2_6)/count
    SWC_1_2_A = SWC_1_2_A.rename("SWC_1_2_A")

# -----STAMP Data. Ameriflux data variable assigned for consistency.------

    # Soil temperature from the west profile 5 cm depth.
    try:
        TS_3_1_1 = ds['soil_temperature_west'][:, 0].rename(
            'TS_3_1_1').drop_vars('depth')
    except KeyError:
        TS_3_1_1 = missing_var("TS_3_1_1")

    # Soil temperatuer from the west profile for 10 cm depth.
    try:
        TS_3_2_1 = ds['soil_temperature_west'][:, 1].rename(
            'TS_3_2_1').drop_vars('depth')
    except KeyError:
        TS_3_2_1 = missing_var("TS_3_2_1")

    # Soil temperatuer from the west profile for 20 cm depth.
    try:
        TS_3_3_1 = ds['soil_temperature_west'][:, 2].rename(
            'TS_3_3_1').drop_vars('depth')
    except KeyError:
        TS_3_3_1 = missing_var("TS_3_3_1")
        
    # Soil temperatuer from the west profile for 50 cm depth.
    try:
        TS_3_4_1 = ds['soil_temperature_west'][:, 3].rename(
            'TS_3_4_1').drop_vars('depth')
    except KeyError:
        TS_3_4_1 = missing_var("TS_3_4_1")

    # Soil temperatuer from the west profile for 100 cm depth.
    try:
        TS_3_5_1 = ds['soil_temperature_west'][:, 4].rename(
            'TS_3_5_1').drop_vars('depth')
    except KeyError:
        TS_3_5_1 = missing_var("TS_3_5_1")

    # Soil temperature from the eat profile 5 cm depth.
    try:
        TS_4_1_1 = ds['soil_temperature_east'][:, 0].rename(
            'TS_4_1_1').drop_vars('depth')
    except KeyError:
        TS_4_1_1 = missing_var("TS_4_1_1")

    # Soil temperatuer from the east profile for 10 cm depth.
    try:
        TS_4_2_1 = ds['soil_temperature_east'][:, 1].rename(
            'TS_4_2_1').drop_vars('depth')
    except KeyError:
        TS_4_2_1 = missing_var("TS_4_2_1")

    # Soil temperatuer from the east profile for 20 cm depth.
    try:
        TS_4_3_1 = ds['soil_temperature_west'][:, 2].rename(
            'TS_4_3_1').drop_vars('depth')
    except KeyError:
        TS_4_3_1 = missing_var("TS_4_3_1")

    # Soil temperatuer from the east profile for 50 cm depth.
    try:
        TS_4_4_1 = ds['soil_temperature_west'][:, 3].rename(
            'TS_4_4_1').drop_vars('depth')
    except KeyError:
        TS_4_4_1 = missing_var("TS_4_4_1")
        
    # Soil temperatuer from the east profile for 100 cm depth.
    try:
        TS_4_5_1 = ds['soil_temperature_west'][:, 4].rename(
            'TS_4_5_1').drop_vars('depth')
    except KeyError:
        TS_4_5_1 = missing_var("TS_4_5_1")

    # Soil temperature from the south profile 5 cm depth.
    try:
        TS_5_1_1 = ds['soil_temperature_south'][:, 0].rename(
            'TS_5_1_1').drop_vars('depth')
    except KeyError:
        TS_5_1_1 = missing_var("TS_5_1_1")

    # Soil temperatuer from the south profile for 10 cm depth.
    try:
        TS_5_2_1 = ds['soil_temperature_south'][:, 1].rename(
            'TS_5_2_1').drop_vars('depth')
    except KeyError:
        TS_5_2_1 = missing_var("TS_5_2_1")

    # Soil temperatuer from the south profile for 20 cm depth.
    try:
        TS_5_3_1 = ds['soil_temperature_south'][:, 2].rename(
            'TS_5_3_1').drop_vars('depth')
    except KeyError:
        TS_5_3_1 = missing_var("TS_5_3_1")

    # Soil temperatuer from the west profile for 50 cm depth.
    try:
        TS_5_4_1 = ds['soil_temperature_south'][:, 3].rename(
            'TS_5_4_1').drop_vars('depth')
    except KeyError:
        TS_5_4_1 = missing_var("TS_5_4_1")

    # Soil temperatuer from the south profile for 100 cm depth.
    try:
        TS_5_5_1 = ds['soil_temperature_south'][:, 4].rename(
            'TS_5_5_1').drop_vars('depth')
    except KeyError:
        TS_5_5_1 = missing_var("TS_5_5_1")


# -----STAMPPCP Data. Ameriflux data variable assigned for consistency.------

    # 30 minute averaged precipitation data in mm.
    try:
        P = ds['precip'].rename('P')
    except KeyError:
        P = missing_var("P")

    # This creates an entire dataframe subset of the variables created above.

    df_subset = [TIMESTAMP_START, TIMESTAMP_END, FC, CO2, CO2_MIXING_RATIO,
                 H2O, H2O_MIXING_RATIO, CH4, CH4_MIXING_RATIO, TAU, H, LE, TA,
                 PA, RH, T_SONIC, VPD, MO_LENGTH, ZL, WS, WD, USTAR, WS_MAX,
                 SW_IN, SW_OUT, LW_IN, LW_OUT, ALB, NETRAD, G_1_1_1, G_1_1_2,
                 G_1_1_3, G_1_1_A, TS_1_1_1, TS_1_1_2, TS_1_1_3, TS_1_1_A,
                 TS_2_1_1, TS_2_1_2, TS_2_1_3, TS_2_1_4, TS_2_1_5, TS_2_1_6,
                 TS_2_1_A, TS_2_2_1, TS_2_2_2, TS_2_2_3, TS_2_2_4, TS_2_2_5,
                 TS_2_2_6, TS_2_2_A, PPFD_IN, PPFD_OUT, SWC_1_1_1, SWC_1_1_2,
                 SWC_1_1_3, SWC_1_1_4, SWC_1_1_5, SWC_1_1_6, SWC_1_1_A,
                 SWC_1_2_1, SWC_1_2_2, SWC_1_2_3, SWC_1_2_4, SWC_1_2_5,
                 SWC_1_2_6, SWC_1_2_A, P, TS_3_1_1, TS_3_2_1, TS_3_3_1,
                 TS_3_4_1, TS_3_5_1, TS_4_1_1, TS_4_2_1, TS_4_3_1, TS_4_4_1,
                 TS_4_5_1, TS_5_1_1, TS_5_2_1, TS_5_3_1, TS_5_4_1, TS_5_5_1, ]

    ds_out = xr.Dataset()
    for dataarray in df_subset:
        ds_out = xr.merge([ds_out, dataarray.to_dataset()])

    df_out = ds_out.to_dataframe()

    df_out.to_csv(Ameriflux_name + '_'+'HH_'+str(df_out['TIMESTAMP_START'][0]) 
                  + '_' + str(df_out['TIMESTAMP_END'][47])+'.csv', index=False)
