# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 20:26:26 2021

@author: T Y van der Duim
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ['PROJ_LIB'] = r"C:\Users\fafri\anaconda3\pkgs\proj4-6.1.1-hc2d0af5_1\Library\share"
from mpl_toolkits.basemap import Basemap
import numpy as np
from datetime import date, timedelta, datetime
import matplotlib.animation as animation
from netCDF4 import Dataset
from pyhdf.SD import SD, SDC
import json
import cdsapi

#%%
'''
---------------------------CERES-MODIS L3 product------------------------------
'''

class hdf4_files:
    '''
    adj_cld_od (97): adjusted cloud optical depth
    adj_uth (91): adjusted upper tropospheric relative humidity
    adj_cld_amount (?): adjusted cloud amount %
    adj_cld_temp (96): adjusted cloud temperature (K)
    '''
    def __init__(self, file_name):
        self.file_name = file_name
        
    def import_hdf4(self):
        file = SD(self.file_name, SDC.READ)
        
        datasets_dic = file.datasets()
        
        for idx,sds in enumerate(datasets_dic.keys()):
            print(idx,sds)
            sds_obj = file.select(sds)
        
            for key, value in sds_obj.attributes().items():
                print("{0}: {1}".format(key, value))
        return file
    
    def select_var(self, var):
        sds_obj = self.import_hdf4().select(var) # select variable
    
        data = sds_obj.get() # get sds data
        print("SELECTED PARAMETER")
        for key, value in sds_obj.attributes().items():
            print("{0}: {1}".format(key, value))
            if key == "_FillValue":
                fillvalue = value
            if key == 'add_offset':
                add_offset = value
            elif key == 'scale_factor':
                scale_factor = value
            else:
                add_offset = 0
                scale_factor = 1
        print('add_offset: ', add_offset)
        print('scale_factor: ', scale_factor)
        
        data = np.where(data == fillvalue, np.nan, data)
        return data

file_name = 'C:\\Users\\fafri\Documents\\Python Scripts\\CER_SYN1deg-1Hour_Terra-Aqua-MODIS_Edition4A_401405.20150301.hdf'

ceres_03_15 = hdf4_files(file_name)

cloud_cover = ceres_03_15.select_var('adj_cld_amount')[0]
cloud_od = ceres_03_15.select_var('adj_cld_od')[0]

plt.figure()
plt.hist(np.ravel(cloud_od), bins = 10000)

#%%
# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])

fig = plt.figure(figsize=(16,12))
ax = plt.subplot(111)
plt.rcParams.update({'font.size': 15})#-- create map
map = Basemap(projection='cyl',llcrnrlat= 35.,urcrnrlat= 60.,\
              resolution='l',  llcrnrlon=-10.,urcrnrlon=40.)
#-- draw coastlines and country boundaries, edge of map
map.drawcoastlines()
map.drawcountries()
map.bluemarble()
#-- create and draw meridians and parallels grid lines
map.drawparallels(np.arange( -90., 90.,30.),labels=[1,0,0,0],fontsize=10)
map.drawmeridians(np.arange(-180.,180.,30.),labels=[0,0,0,1],fontsize=10)

lon, lat = map(*np.meshgrid(np.arange(-10,41,1),np.arange(35,61,1)))

# contourf 
i = 0
im = map.contourf(lon, lat, cloud_cover[0, i, 125:151, 170:221], np.arange(0, 100, 0.1), extend='max', cmap='jet')
cbar=plt.colorbar(im)
date = '01-03-2015 00:00:00'
time = datetime.strptime(date, '%d-%m-%Y %H:%M:%S')
title = ax.text(0.5,1.05,time,
                    ha="center", transform=ax.transAxes,)
 
def animate(i):
    global im, title
    for c in im.collections:
        c.remove()
    title.remove()
    im = map.contourf(lon, lat, cloud_cover[0, i, 125:151, 170:221], np.arange(0, 100, 0.1), extend='max', cmap='jet')
    date = '01-03-2015 00:00:00'
    time = datetime.strptime(date, '%d-%m-%Y %H:%M:%S') + timedelta(hours = i)
    title = ax.text(0.5,1.05,"cirrus cover from NASA Satellite Data\n{0}".format(time),
                    ha="center", transform=ax.transAxes,)

    
myAnimation = animation.FuncAnimation(fig, animate, frames = 24)
myAnimation.save('cirruscoverNASA.mp4', writer=writer)

#%%
'''
------------------ERA5 reanalysis data on single levels----------------------
'''

def api_req():
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': 'high_cloud_cover',
            'year': '2015',
            'month': '03',
            'day': [
                '01', '02', '03',
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': [
                60, -10, 35,
                40,
            ],
        },
        'era5singleL20150301-03.nc')

api_req()

#%%

def netcdf_import(file):
    data = Dataset(file,'r')
    print(data.variables)
    print(data.variables.keys())
    # for key, value in data.variables.items():
    #     if key == "_FillValue":
    #         fillvalue = value
    #         print(fillvalue)
        
        # data = np.where(data == fillvalue, np.nan, data)
    my_dict = {}
    for key in data.variables.keys():
        if key == 'longitude' or key == 'latitude' or key == 'time' or key == 'level':
            my_dict[key] = data[key][:]
        else:
            my_dict[key] = data[key][:, :, :]
    return my_dict

era5singleL2015030103 = netcdf_import('era5singleL20150301-03.nc')

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])

fig = plt.figure(figsize=(16,12))
ax = plt.subplot(111)
plt.rcParams.update({'font.size': 15})#-- create map
map = Basemap(projection='cyl',llcrnrlat= 35.,urcrnrlat= 60.,\
              resolution='l',  llcrnrlon=-10.,urcrnrlon=40.)
#-- draw coastlines and country boundaries, edge of map
map.drawcoastlines()
map.drawcountries()
map.bluemarble()

#-- create and draw meridians and parallels grid lines
map.drawparallels(np.arange( -90., 90.,30.),labels=[1,0,0,0],fontsize=10)
map.drawmeridians(np.arange(-180.,180.,30.),labels=[0,0,0,1],fontsize=10)

lons, lats = map(*np.meshgrid(era5singleL2015030103['longitude'], era5singleL2015030103['latitude']))

# contourf 
im = map.contourf(lons, lats, era5singleL2015030103['hcc'][0], np.arange(0, 1.01, 0.01), extend='neither', cmap='jet')
cbar=plt.colorbar(im)
date = '01-03-2015 00:00:00'
time = datetime.strptime(date, '%d-%m-%Y %H:%M:%S')
title = ax.text(0.5,1.05,time,
                    ha="center", transform=ax.transAxes,)
 
def animate(i):
    global im, title
    for c in im.collections:
        c.remove()
    title.remove()
    im = map.contourf(lons, lats, era5singleL2015030103['hcc'][i], np.arange(0, 1.01, 0.01), extend='neither', cmap='jet')
    date = '01-03-2015 00:00:00'
    time = datetime.strptime(date, '%d-%m-%Y %H:%M:%S') + timedelta(hours = i)
    title = ax.text(0.5,1.05,"cirrus cover from ERA5 Reanalysis Data\n{0}".format(time),
                    ha="center", transform=ax.transAxes,)
    
myAnimation = animation.FuncAnimation(fig, animate, frames = 72)
myAnimation.save('cirruscoverERA5.mp4', writer=writer)

#%%
'''
------------------ERA5 reanalysis data on pressure levels----------------------
'''

def api_req():
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'relative_humidity', 'specific_cloud_ice_water_content', 'specific_cloud_liquid_water_content',
                'temperature', 'u_component_of_wind', 'v_component_of_wind',
            ],
            'pressure_level': [
                '100', '125', '150',
                '175', '200', '225',
                '250', '300', '350',
            ],
            'year': '2015',
            'month': '03',
            'day': [
                '01', '02', '03',
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': [
                60, -10, 35,
                40,
            ],
        },
        'era5pressureL20150301-03.nc')

api_req()

#%%

def netcdf_import(file, *args):
    data = Dataset(file,'r')
    print(data.variables)
    pres_level = args[0]
    print(data.variables.keys())
    # for key, value in data.variables.items():
    #     if key == "_FillValue":
    #         fillvalue = value
    #         print(fillvalue)
        
        # data = np.where(data == fillvalue, np.nan, data)
    my_dict = {}
    if 'level' in data.variables.keys():
        idx = np.where(data['level'][:] == pres_level)[0][0]
    for key in data.variables.keys():
        if key == 'longitude' or key == 'latitude' or key == 'time' or key == 'level':
            my_dict[key] = data[key][:]
        else:
            my_dict[key] = data[key][:, idx, :, :]
    return my_dict

era5pressureL2015030103 = netcdf_import('era5pressureL20150301-03.nc', 250)


# Set up formatting for the movie files
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])

fig = plt.figure(figsize=(16,12))
ax = plt.subplot(111)
plt.rcParams.update({'font.size': 15})#-- create map
map = Basemap(projection='cyl',llcrnrlat= 35.,urcrnrlat= 60.,\
              resolution='l',  llcrnrlon=-10.,urcrnrlon=40.)
#-- draw coastlines and country boundaries, edge of map
map.drawcoastlines()
map.drawcountries()
map.bluemarble()

#-- create and draw meridians and parallels grid lines
map.drawparallels(np.arange( -90., 90.,30.),labels=[1,0,0,0],fontsize=10)
map.drawmeridians(np.arange(-180.,180.,30.),labels=[0,0,0,1],fontsize=10)

lons, lats = map(*np.meshgrid(era5pressureL2015030103['longitude'], era5pressureL2015030103['latitude']))

# contourf 
im = map.contourf(lons, lats, era5pressureL2015030103['r'][0], np.arange(0, 100.1, 0.1), extend='both', cmap='jet')
cbar=plt.colorbar(im)
date = '01-03-2015 00:00:00'
time = datetime.strptime(date, '%d-%m-%Y %H:%M:%S')
title = ax.text(0.5,1.05,time,
                    ha="center", transform=ax.transAxes,)
 
def animate(i):
    global im, title
    for c in im.collections:
        c.remove()
    title.remove()
    im = map.contourf(lons, lats, era5pressureL2015030103['r'][i], np.arange(0, 100.1, 0.1), extend='both', cmap='jet')
    date = '01-03-2015 00:00:00'
    time = datetime.strptime(date, '%d-%m-%Y %H:%M:%S') + timedelta(hours = i)
    title = ax.text(0.5,1.05,"RH from ERA5 Reanalysis Data\n{0}".format(time),
                    ha="center", transform=ax.transAxes,)
    
myAnimation = animation.FuncAnimation(fig, animate, frames = 72)
#myAnimation.save('RHRean.mp4', writer=writer)
