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
from scipy import stats

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

res_lon = 0.25 # set desired resolution in longitudinal direction in degs
res_lat = 0.25 # set desired resolution in latitudinal direction in degs
min_lon = -10
max_lon = 40
min_lat = 35
max_lat = 60

file = 'C:\\Users\\fafri\\Documents\\Python Scripts\\ORD42131\\CM_SAF_CLAAS2_L2_AUX.nc'
auxfile = Dataset(file,'r')

def L2_to_L3(res_lon, res_lat):
    lon_lat_grid = stats.binned_statistic_2d(lon.flatten(), lat.flatten(),
                                     ct_cirrus[0].flatten(), statistic = 'mean',
                                     bins = [int((max_lon - min_lon) / res_lon), int((max_lat - min_lat) / res_lat)],
                                     range = [[min_lon, max_lon], [min_lat, max_lat]])
    
    return lon_lat_grid.statistic

ct_day = []

for filename in os.listdir('C:\\Users\\fafri\\Documents\\Python Scripts\\ORD42131'):
    #file = 'C:\\Users\\fafri\\Documents\\Python Scripts\\ORD42131\\{0}.nc'.format(name_iter)
    if filename.endswith("UD.nc"):
        print(filename)
        L2_data = Dataset('C:\\Users\\fafri\\Documents\\Python Scripts\\ORD42131\\' + str(filename),'r')
        
        # extract variables of interest
        ct = L2_data['ct'][:]
        cot = L2_data['cot'][:]
        
        # filter out cirrus clouds
        ct_cirrus = np.where(ct == 7, 1, 0) # all cirrus occurrences 1, the rest 0
        cot_cirrus = np.where(ct_cirrus == 1, cot, np.nan) # all non-cirrus pixels NaN
        cot_cirrus[cot_cirrus == -1] = np.nan # make all invalid data (-1) NaN
        
        # coordinates
        lat = auxfile['lat'][:]
        lon = auxfile['lon'][:]
            
        ct_day.append(L2_to_L3(res_lon, res_lat))
        
ct_day = np.stack(ct_day)

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

lons, lats = map(*np.meshgrid(np.arange(min_lon + res_lon/2, max_lon, res_lon),
                              np.arange(min_lat + res_lat/2, max_lat, res_lat)))

# contourf 
im = map.contourf(lons, lats, ct_day[0].T, np.arange(0, 1.01, 0.01), extend='neither', cmap='jet')
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
    im = map.contourf(lons, lats, ct_day[i].T, np.arange(0, 1.01, 0.01), extend='neither', cmap='jet')
    date = '28-02-2015 00:00:00'
    time = datetime.strptime(date, '%d-%m-%Y %H:%M:%S') + timedelta(minutes = i * 15)
    title = ax.text(0.5,1.05,"Cloud cover from SEVIRI Meteosat CLAAS\n{0}".format(time),
                    ha="center", transform=ax.transAxes,)
    
myAnimation = animation.FuncAnimation(fig, animate, frames = len(ct_day))
myAnimation.save('cirruscoverCLAAS.mp4', writer=writer)
