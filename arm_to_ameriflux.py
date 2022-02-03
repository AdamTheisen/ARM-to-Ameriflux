import act
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import xarray as xr
import json
import os

#Read in ARM Live Data Webservice Token and Username
with open('./token.json') as f:
    data = json.load(f)
username = data['username']
token = data['token']

# Set the facility and instruments to use
facs = ['E39']
datastreams = ['ecorsf', 'sebs', 'amc']

# Set dates, first one is to use for downloading data, second is ARM format
date = '2021-05-01'
sdate = ''.join(date.split('-'))
for f in facs:
    ecor = None
    sebs = None
    amc = None
    obj = []

    # Look for ECOR data, download if None, and read in
    ecor_files = glob.glob('./sgpecorsf' + f + '.b1/*' + sdate + '*')
    if len(ecor_files) == 0:
        ecor_files = act.discovery.download_data(username, token, 'sgpecorsf' + f + '.b1', date, date)
    if len(ecor_files) > 0:
        ecor =  act.io.armfiles.read_netcdf(ecor_files)
        ecor = ecor.rename({'time_bounds': 'time_bounds_ecor'})
        # Add to object for merging datasets at the end
        obj.append(ecor)

    # Look for SEBS data, download if None, and read in
    sebs_files = glob.glob('./sgpsebs' + f + '.b1/*' + sdate + '*')
    if len(sebs_files) == 0:
        sebs_files = act.discovery.download_data(username, token, 'sgpsebs' + f + '.b1', date, date)
    if len(sebs_files) > 0:
        sebs =  act.io.armfiles.read_netcdf(sebs_files)
        obj.append(sebs)

    # Look for AMC data, download if None, and read in
    amc_files = glob.glob('./sgpamc' + f + '.b1/*' + sdate + '*')
    if len(amc_files) == 0:
        amc_files = act.discovery.download_data(username, token, 'sgpamc' + f + '.b1', date, date)
    if len(amc_files) > 0:
        amc =  act.io.armfiles.read_netcdf(amc_files)
        amc = amc.rename({'time_bounds': 'time_bounds_amc'})
        obj.append(amc)

    # Merge 3 instruments together into one xarray object
    obj = xr.merge(obj, compat='override')

    # Close out individual objects
    ecor.close()
    sebs.close()
    amc.close()

    # Create dirs and write out merged data
    if not os.path.exists('./sgpflux' + f + '.b1/'):
        os.mkdir('./sgpflux' + f + '.b1/')
    obj.to_netcdf('./sgpflux' + f + '.b1/sgpflux' + f + '.b1.' + sdate + '.000000.nc')
