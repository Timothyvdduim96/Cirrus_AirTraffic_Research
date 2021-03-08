# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:46:25 2021

@author: T Y van der Duim
"""

'''
------------------------------PACKAGES-----------------------------------------
'''

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
from requests import Session
import sys
import urllib
import fnmatch
import lxml.html
import wget
import keyring
from scipy import stats
import cdsapi

'''
---------------------------PLOTTING PREFERENCES--------------------------------
'''

plt.style.use('seaborn-darkgrid')
plt.rc('text', usetex=False)
plt.rc('font', family='times')
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 
plt.rc('font', size=20) 
plt.rc('figure', figsize = (12, 5))

#%%
'''
-----------------------------PROCESS HDF4 DATA---------------------------------
'''

class hdf4_files:

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
    
#%%

'''
-----------------------------PROCESS ERA5 DATA---------------------------------
'''

path = "C:\\Users\\fafri\\Documents\\ads_cirrus_airtraffic\\ERA5_data"

def netcdf_import(file, *args):
    data = Dataset(file,'r')
    print(data.variables)
    pres_level = args[0]
    print(data.variables.keys())
    my_dict = {}
    if 'level' in data.variables.keys():
        idx = np.where(data['level'][:] == pres_level)[0][0]
    for key in data.variables.keys():
        if key == 'longitude' or key == 'latitude' or key == 'time' or key == 'level':
            my_dict[key] = data[key][:]
        else:
            my_dict[key] = data[key][:, idx, :, :]
    return my_dict

ERA5dict = netcdf_import(path + '\\ERA5_15.nc', 250)

def animate(dataset_dict, var):
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
    
    lons, lats = map(*np.meshgrid(dataset_dict['longitude'], dataset_dict['latitude']))
    
    # contourf 
    im = map.contourf(lons, lats, dataset_dict[var][0], np.arange(0, 100.1, 0.1), extend='both', cmap='jet')
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
        im = map.contourf(lons, lats, dataset_dict[var][i], np.arange(0, 100.1, 0.1), extend='both', cmap='jet')
        date = '01-03-2015 00:00:00'
        time = datetime.strptime(date, '%d-%m-%Y %H:%M:%S') + timedelta(hours = i)
        title = ax.text(0.5,1.05,"RH from ERA5 Reanalysis Data\n{0}".format(time),
                        ha="center", transform=ax.transAxes,)
        
    myAnimation = animation.FuncAnimation(fig, animate, frames = 72)

#%%

'''
-----------------------PROCESS METEOSAT CLAAS 2.1 DATA-------------------------
'''

res_lon = 0.25 # set desired resolution in longitudinal direction in degs
res_lat = 0.25 # set desired resolution in latitudinal direction in degs
min_lon = -10 # research area min longitude
max_lon = 40 # research area max longitude
min_lat = 35 # research area min latitude
max_lat = 60 # research area max latitude

path = 'C:\\Users\\fafri\\Documents\\ads_cirrus_airtraffic\\Meteosat_CLAAS_data'

file = '{0}\\CM_SAF_CLAAS2_L2_AUX.nc'.format(path)
auxfile = Dataset(file,'r')

def L2_to_L3(var, res_lon, res_lat, stat):
    lon_lat_grid = stats.binned_statistic_2d(lon.flatten(), lat.flatten(),
                                     var[0].flatten(), statistic = stat,
                                     bins = [int((max_lon - min_lon) / res_lon), int((max_lat - min_lat) / res_lat)],
                                     range = [[min_lon, max_lon], [min_lat, max_lat]])
    
    return lon_lat_grid.statistic

ct_sample = []
#cot_sample = []

def nanmax(array):
    return np.nanmax(array)

for filename in os.listdir(path):
    #file = 'C:\\Users\\fafri\\Documents\\Python Scripts\\ORD42131\\{0}.nc'.format(name_iter)
    if filename.endswith("UD.nc"):
        print(filename)
        L2_data = Dataset(path + "\\" + str(filename),'r')
        
        # extract variables of interest
        ct = L2_data['ct'][:]
        cot = L2_data['cot'][:]
        
        # filter out cirrus clouds
        ct_cirrus = np.where(ct == 7, 1, 0) # all cirrus occurrences 1, the rest 0
        cot_cirrus = np.where(ct_cirrus == 1, cot, np.nan) # all non-cirrus pixels NaN
        cot_cirrus = np.where((cot_cirrus == -1) | (cot_cirrus == np.nan), np.nan, cot_cirrus) # make all invalid data (-1) NaN
        
        # coordinates
        lat = auxfile['lat'][:]
        lon = auxfile['lon'][:]
            
        ct_sample.append(L2_to_L3(ct_cirrus, res_lon, res_lat, stat = 'mean'))
        #cot_sample.append(L2_to_L3(cot_cirrus, res_lon, res_lat, stat = lambda x: nanmax(x)))
        
ct_sample = np.stack(ct_sample)
ct_sample = np.where(ct_sample == 0, np.nan, ct_sample)
#cot_sample = np.stack(cot_sample)

# Set up formatting for the movie files
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])

fig = plt.figure(figsize=(16,12))
ax = plt.subplot(111)
plt.rcParams.update({'font.size': 15})#-- create map
map = Basemap(projection='cyl',llcrnrlat= 35 + res_lat/2, urcrnrlat= 60 - res_lat/2,\
              resolution='l',  llcrnrlon=-10 + res_lon/2, urcrnrlon=40 - res_lon/2)
#-- draw coastlines and country boundaries, edge of map
map.drawcoastlines()
map.drawcountries()
map.bluemarble()

#-- create and draw meridians and parallels grid lines
map.drawparallels(np.arange( -90., 90.,15.),labels=[1,0,0,0],fontsize=10)
map.drawmeridians(np.arange(-180.,180.,15.),labels=[0,0,0,1],fontsize=10)

lons, lats = map(*np.meshgrid(np.arange(min_lon + res_lon/2, max_lon, res_lon),
                              np.arange(min_lat + res_lat/2, max_lat, res_lat)))

# contourf 
im = map.contourf(lons, lats, ct_sample[0].T, np.arange(0, 1.01, 0.01), 
                  extend='neither', cmap='binary')
cbar=plt.colorbar(im)
date = '28-02-2015 00:00:00'
time = datetime.strptime(date, '%d-%m-%Y %H:%M:%S')
title = ax.text(0.5,1.05,time,
                    ha="center", transform=ax.transAxes,)

def animate(i):
    global im, title
    for c in im.collections:
        c.remove()
    title.remove()
    im = map.contourf(lons, lats, ct_sample[i].T, np.arange(0, 1.01, 0.01), 
                      extend='neither', cmap='binary')
    date = '11-03-2015 00:00:00'
    time = datetime.strptime(date, '%d-%m-%Y %H:%M:%S') + timedelta(minutes = i * 15)
    title = ax.text(0.5,1.05,"Cloud cover from SEVIRI Meteosat CLAAS\n{0}".format(time),
                    ha="center", transform=ax.transAxes,)
    
myAnimation = animation.FuncAnimation(fig, animate, frames = len(ct_sample))
#myAnimation.save('cirruscoverCLAAS.mp4', writer=writer)




