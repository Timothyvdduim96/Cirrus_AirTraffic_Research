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
from scipy import stats
from datetime import time
import time

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

'''
--------------------------------ROI--------------------------------------------
'''

res_lon = 0.25 # set desired resolution in longitudinal direction in degs
res_lat = 0.25 # set desired resolution in latitudinal direction in degs
min_lon = -10 # research area min longitude
max_lon = 40 # research area max longitude
min_lat = 35 # research area min latitude
max_lat = 60 # research area max latitude

def L2_to_L3(var, dim1, dim2, res_lon, res_lat, stat):
    lon_lat_grid = stats.binned_statistic_2d(dim1.flatten(), dim2.flatten(),
                                     var.flatten(), statistic = stat,
                                     bins = [int((max_lon - min_lon) / res_lon), int((max_lat - min_lat) / res_lat)],
                                     range = [[min_lon, max_lon], [min_lat, max_lat]])
    
    return lon_lat_grid.statistic

#%%
'''
----------------------------PROCESS CALIPSO DATA-------------------------------
'''
path = "E:\\Research_cirrus\\CALIPSO_data\\"

class hdf4_files:

    def __init__(self, file_name):
        self.file_name = file_name
        
    def import_hdf4(self):
        file = SD(self.file_name, SDC.READ)
        
        datasets_dic = file.datasets()
        
        for idx,sds in enumerate(datasets_dic.keys()):
            #print(idx,sds)
            sds_obj = file.select(sds)
        
            #for key, value in sds_obj.attributes().items():
                #print("{0}: {1}".format(key, value))
        return file
    
    def select_var(self, var):
        sds_obj = self.import_hdf4().select(var) # select variable
    
        data = sds_obj.get() # get sds data
        #print("SELECTED PARAMETER")
        for key, value in sds_obj.attributes().items():
            #print("{0}: {1}".format(key, value))
            if key == "_FillValue":
                fillvalue = value
            if key == 'add_offset':
                add_offset = value
            elif key == 'scale_factor':
                scale_factor = value
            else:
                add_offset = 0
                scale_factor = 1
        #print('add_offset: ', add_offset)
        #print('scale_factor: ', scale_factor)
        try:
            data = np.where(data == fillvalue, np.nan, data)
        except:
            data = data
        return data

cirrus_cover = pd.DataFrame()
cirrus_ani = []
dates = []
times = []

def rounder(t):
    t = datetime.strptime(t, '%H:%M:%S')
    return t.round('H')
    
# convert binary numbers into info on sample
def feature_class(var, start, end):
    fclass = []
    fillvalue = -999
    select_var = {1: feature_type, 2: feature_QA, 3: feature_subcloud}
    var_dict = select_var[var]
    
    for idx, flag in enumerate(binnrs):
        try:
            fclass.append(var_dict[int(flag[start:end], 2)])
        except:
            fclass.append(fillvalue)
        
    return fclass

def convert_seconds(n): 
    return time.strftime('%H:%M:%S', time.gmtime(n))

for filename in os.listdir(path + 'LIDAR_03_15\\'):
    print(path + 'LIDAR_03_15\\' + str(filename))
    test_calipso = hdf4_files(path + 'LIDAR_03_15\\' + str(filename))
    
    class_flag = test_calipso.select_var("Feature_Classification_Flags")
    t = test_calipso.select_var("Profile_UTC_Time")
    lat = test_calipso.select_var("Latitude")
    lon = test_calipso.select_var("Longitude")
    
    # feature_subcloud = {0: 'low overcast, transparent', 1: 'low overcast, opaque',
    #                 2: 'transition stratocumulus', 3: 'low, broken cumulus',
    #                4: 'altocumulus (transparent)', 5: 'altostratus (opaque)',
    #                6: 'cirrus (transparent)', 7: 'deep convective (opaque)'}
    
    feature_type = {0: 'invalid', 1: 'clear air', 2: 'cloud', 3: 'aerosol', 4: 'stratospheric feature',
            5: 'surface', 6: 'subsurface', 7: 'nosignal'}
    
    feature_QA = {0: 'none', 1: 'low', 2: 'medium', 3: 'high'}
    
    feature_subcloud = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 
                        7: '7'}
    
    binnrs = [format(flag, 'b') for flag in np.unique(class_flag)] # convert unique classification flags into binary
    
    feature_type_data = feature_class(1, -3, None)
    feature_QA_data = feature_class(2, -5, -3)
    feature_subtype_data = feature_class(3, -12, -9)
    
    features = pd.DataFrame({'type': feature_type_data,
                             'QA': feature_QA_data,
                             'subtype': feature_subtype_data}, index = np.unique(class_flag))
    
    # mark cirrus detections with high or middle confidence 1, no cirrus detection 0
    # and cirrus detection with low confidence -1
    
    conditions = [
        (features['type'] == 'cloud') & ((features['QA'] == 'medium') | (features['QA'] == 'high'))
        & (features['subtype'] == '6'), (features['type'] != 'cloud') | ((features['type'] == 'cloud')
        & (features['subtype'] != '6')), (features['type'] == 'cloud') &
        (features['QA'] == 'low') & (features['subtype'] == '6')]
    
    fills = [1, 0, np.nan]
    
    features['cirrus'] = np.select(conditions, fills)
    
    feature_dict = pd.Series(features['cirrus'], index = features.index).to_dict() # convert feature flag and cirrus (1 or 0) to dictionary
    cirrus_flag = pd.DataFrame(class_flag).replace(feature_dict) # map this coding to classification flag array
    cirrus_flag = cirrus_flag.apply(lambda x: max(x), axis = 1) # convert to 1D (remove multi-layered case for ease of analysis)
    #cirrus_flag = cirrus_flag.astype(int)
    
    # format time
    t = [str(t_instance[0]).split('.') for t_instance in t] # split date from time
    t = pd.DataFrame(t, columns = ['date', 'time'])
    t['date'] = pd.to_datetime(t['date'], format = '%y%m%d').dt.date
    t['time'] = t['time'].apply(lambda x: 24 * 3600 * eval('0.{0}'.format(x)))
    t['time'] = t['time'].apply(lambda x: convert_seconds(x))
    
    # gather data and select relevant data (within ROI)
    df_cirrus = pd.concat([t, pd.DataFrame(lon), pd.DataFrame(lat), cirrus_flag], axis = 1)
    df_cirrus.columns = ['date', 'time', 'lon', 'lat', 'cirrus_cover']
    
    df_cirrus = df_cirrus[(df_cirrus['lon'] >= min_lon) & (df_cirrus['lon'] <= max_lon)
                          & (df_cirrus['lat'] >= min_lat) & (df_cirrus['lat'] <= max_lat)]
    
    cirrus_gridded = L2_to_L3(np.array(df_cirrus['cirrus_cover']), np.array(df_cirrus['lon']),
             np.array(df_cirrus['lat']), res_lon, res_lat, 'mean')
    
    cirrus_ani.append(cirrus_gridded)
    dates.append(df_cirrus['date'].iloc[0])
    times.append(df_cirrus['time'].iloc[int(len(df_cirrus) / 2)][:3] + str("00"))
    #cirrus_cover = pd.concat([cirrus_cover, df_cirrus])

#%%
cirrus_ani = np.stack(cirrus_ani)

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])

#def animate(dataset_dict, var):
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
map.drawparallels(np.arange( -90., 90.,10.),labels=[1,0,0,0],fontsize=10)
map.drawmeridians(np.arange(-180.,180.,10.),labels=[0,0,0,1],fontsize=10)

lons, lats = map(*np.meshgrid(np.arange(-10, 40, 0.25), np.arange(35, 60, 0.25)))

# contourf 
im = map.contourf(lons, lats, cirrus_ani[0].T, np.arange(0, 1.01, 0.01), extend='neither', cmap='jet')
cbar=plt.colorbar(im)
#date = '01-03-2015 00:00:00'
#time = datetime.strptime(date, '%d-%m-%Y %H:%M:%S')
title = ax.text(0.5,1.05,dates[0],
                    ha="center", transform=ax.transAxes,)

def animate(i):
    global im, title
    for c in im.collections:
        c.remove()
    title.remove()
    im = map.contourf(lons, lats, cirrus_ani[i].T, np.arange(0, 1.01, 0.01), extend='neither', cmap='jet')
    #date = '01-03-2015 00:00:00'
    #time = datetime.strptime(date, '%d-%m-%Y %H:%M:%S') + timedelta(hours = i)
    title = ax.text(0.5,1.05,"Cirrus cover from CALIPSO\n{0} {1}".format(dates[i], times[i]),
                    ha="center", transform=ax.transAxes,)
    
myAnimation = animation.FuncAnimation(fig, animate, frames = len(cirrus_ani), interval = 1000)
myAnimation.save('CALIPSO_0315.mp4', writer=writer)


#%%

'''
-----------------------------PROCESS ERA5 DATA---------------------------------
'''

path = "E:\\Research_cirrus\\ERA5_data"

def netcdf_import(file, *args):
    data = Dataset(file,'r')
    print(data.variables)
    try:
        pres_level = args[0]
    except:
        pres_level = np.nan
    print(data.variables.keys())
    my_dict = {}
    if 'level' in data.variables.keys():
        try:
            idx = np.where(data['level'][:] == pres_level)[0][0]
        except:
            print("Define a pressure level!")
            return
    for key in data.variables.keys():
        if key == 'longitude' or key == 'latitude' or key == 'time' or key == 'level':
            my_dict[key] = data[key][:]
        else:
            my_dict[key] = data[key][:, idx, :, :]
    return my_dict

ERA5dict = netcdf_import(path + '\\ERA5_15.nc', 250)

#def animate(dataset_dict, var):
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

lons, lats = map(*np.meshgrid(ERA5dict['longitude'], ERA5dict['latitude']))

# contourf 
im = map.contourf(lons, lats, ERA5dict['r'][0], np.arange(0, 100.1, 0.1), extend='both', cmap='jet')
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
    im = map.contourf(lons, lats, ERA5dict['r'][i], np.arange(0, 100.1, 0.1), extend='both', cmap='jet')
    date = '01-03-2015 00:00:00'
    time = datetime.strptime(date, '%d-%m-%Y %H:%M:%S') + timedelta(hours = i)
    title = ax.text(0.5,1.05,"RH from ERA5 Reanalysis Data\n{0}".format(time),
                    ha="center", transform=ax.transAxes,)
    
myAnimation = animation.FuncAnimation(fig, animate, frames = len(ERA5dict['time']))

#%%

'''
-----------------------PROCESS METEOSAT CLAAS 2.1 DATA-------------------------
'''

path = 'E:\\Research_cirrus\\Meteosat_CLAAS_data\\'

file = '{0}CM_SAF_CLAAS2_L2_AUX.nc'.format(path)
auxfile = Dataset(file,'r')

ct_sample = []
#cot_sample = []

def nanmax(array):
    return np.nanmax(array)

for filename in os.listdir(path):
    #file = 'C:\\Users\\fafri\\Documents\\Python Scripts\\ORD42131\\{0}.nc'.format(name_iter)
    if filename.endswith("UD.nc"):
        print(filename)
        L2_data = Dataset(path + str(filename),'r')
        
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
            
        ct_sample.append(L2_to_L3(ct_cirrus[0], lon, lat, res_lon, res_lat, stat = 'mean'))
        #cot_sample.append(L2_to_L3(cot_cirrus, res_lon, res_lat, stat = lambda x: nanmax(x)))
        if filename == 'CPPin20150302000000305SVMSG01UD.nc':
            break
        
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
    date = '01-03-2015 00:00:00'
    time = datetime.strptime(date, '%d-%m-%Y %H:%M:%S') + timedelta(minutes = i * 15)
    title = ax.text(0.5,1.05,"Cloud cover from SEVIRI Meteosat CLAAS\n{0}".format(time),
                    ha="center", transform=ax.transAxes,)
    
myAnimation = animation.FuncAnimation(fig, animate, frames = len(ct_sample))
#myAnimation.save('cirruscoverCLAAS.mp4', writer=writer)




