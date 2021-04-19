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
import math as m
import warnings
from miscellaneous import miscellaneous

warnings.filterwarnings("ignore", category=RuntimeWarning)

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

#%%
'''
----------------------------PROCESS CALIPSO DATA-------------------------------
'''
lidar_path = "E:\\Research_cirrus\\CALIPSO_data\\"

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

class CALIPSO_analysis: # e.g. test = CALIPSO_analysis(lidar_path + 'LIDAR_03_15', 15, False)
    
    def __init__(self, directory_name, time_res, pressure_axis): # pressure axis is a boolean!
        self.directory_name = directory_name
        self.load_hdf4 = hdf4_files
        self.time_res = time_res
        self.pressure_axis = pressure_axis
        # automatically run essential methods
        self.CALIPSO()
        
    def nanmean(self, arr):
        return np.nanmean(arr)
        
    def L2_to_L3(self, var, dim1, dim2, res_lon, res_lat, stat):
        lon_lat_grid = stats.binned_statistic_2d(dim1.flatten(), dim2.flatten(),
                                     var.flatten(), statistic = stat,
                                     bins = [int((max_lon - min_lon) / res_lon), int((max_lat - min_lat) / res_lat)],
                                     range = [[min_lon, max_lon], [min_lat, max_lat]])
    
        return lon_lat_grid.statistic

    def bin_pressureL(self, var):
        bins_pressureL = stats.binned_statistic(var.flatten(),
                                     np.array([87.5, 112.5, 137.5, 162.5, 187.5, 212.5,
                                              237.5, 275, 325, 375, 425, 475]),
                                     statistic = 'count',
                                     bins = [11],
                                     range = [87.5, 475]) # pressure levels in upper troposphere include 100-400 in steps of 50hPa (see SPACE_import)
    
        return bins_pressureL.statistic
        
    def rounder(self, t):
        self.t = datetime.strptime(t, '%H:%M:%S')
        return self.t.round('H')
        
    # convert binary numbers into info on sample
    def feature_class(self, var, start, end, class_flag):
        # feature_subcloud = {0: 'low overcast, transparent', 1: 'low overcast, opaque',
            #                 2: 'transition stratocumulus', 3: 'low, broken cumulus',
            #                4: 'altocumulus (transparent)', 5: 'altostratus (opaque)',
            #                6: 'cirrus (transparent)', 7: 'deep convective (opaque)'}
            
        feature_type = {0: 'invalid', 1: 'clear air', 2: 'cloud', 3: 'aerosol', 4: 'stratospheric feature',
                    5: 'surface', 6: 'subsurface', 7: 'nosignal'}
            
        feature_QA = {0: 'none', 1: 'low', 2: 'medium', 3: 'high'}
            
        feature_subcloud = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 
                                7: '7'}
        self.fclass = []
        fillvalue = -999
        select_var = {1: feature_type, 2: feature_QA, 3: feature_subcloud}
        var_dict = select_var[var]
        binnrs = [format(flag, 'b') for flag in np.unique(class_flag)] # convert unique classification flags into binary
    
        for idx, flag in enumerate(binnrs):
            try:
                self.fclass.append(var_dict[int(flag[start:end], 2)])
            except:
                self.fclass.append(fillvalue)
            
        return self.fclass
    
    def convert_seconds(self, n): 
        timestr = time.strftime('%H:%M:%S', time.gmtime(n))
        return datetime.strptime(timestr, '%H:%M:%S').time()
        
    def CALIPSO(self):
        
        self.layered_cirrus = []
        self.CALIPSO_cirrus = []
        self.cirrus_od = []
        self.CALIPSO_temp = []
        self.CALIPSO_count = []
        self.dates = []
        self.times = []
        self.lon_pos = []
        self.lat_pos = []
        
        for filename in os.listdir(self.directory_name):
            print(self.directory_name + "\\" + str(filename))
            test_calipso = self.load_hdf4(self.directory_name + "\\" + str(filename))
            
            class_flag = test_calipso.select_var("Feature_Classification_Flags")
            t = test_calipso.select_var("Profile_UTC_Time")
            lat = test_calipso.select_var("Latitude")
            lon = test_calipso.select_var("Longitude")
            midlayer_pres = test_calipso.select_var("Midlayer_Pressure") 
            midlayer_temp = test_calipso.select_var("Midlayer_Temperature")
            #opt_depth = test_calipso.select_var("Feature_Optical_Depth_532") 
                    
            feature_type_data = self.feature_class(1, -3, None, class_flag)
            feature_QA_data = self.feature_class(2, -5, -3, class_flag)
            feature_subtype_data = self.feature_class(3, -12, -9, class_flag)
            
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
            
            # format time
            t = [str(t_instance[0]).split('.') for t_instance in t] # split date from time
            t = pd.DataFrame(t, columns = ['date', 'time'])
            
            df_cirrus = pd.concat([t, pd.DataFrame(lon), pd.DataFrame(lat)], axis = 1)

            df_cirrus.columns = ['date', 'time', 'lon', 'lat']
            
            subset_area = np.array([(df_cirrus['lon'] >= min_lon) & (df_cirrus['lon'] <= max_lon)
                                      & (df_cirrus['lat'] >= min_lat) & (df_cirrus['lat'] <= max_lat)]).reshape(-1)
            
            t['date'] = pd.to_datetime(t['date'], format = '%y%m%d').dt.date
            t['time'] = t['time'].apply(lambda x: 24 * 3600 * eval('0.{0}'.format(x)))
            t['time'] = t['time'].apply(lambda x: self.convert_seconds(x))
            combined_datetime = [datetime.combine(date, time) for date, time in zip(t['date'], t['time'])]
    
            combined_datetime = list(np.array(combined_datetime)[subset_area])
            try:
                start_time_window = combined_datetime[0]
                end_time_window = start_time_window + timedelta(minutes = self.time_res)
                
                while start_time_window < combined_datetime[-1]:
                    
                    time_bool = pd.DataFrame(combined_datetime)[0].between(start_time_window, end_time_window)
                    cirrus_flag = pd.DataFrame(class_flag).replace(feature_dict) # map this coding to classification flag array
                    self.cirrus_midlayer_pres = np.where(cirrus_flag == 1, midlayer_pres, np.nan)
                    #self.opt_depth = pd.DataFrame(np.where(cirrus_flag == 1, opt_depth, 0))

                    # if self.pressure_axis == False:
                        
                    #     cirrus_flag = cirrus_flag.apply(lambda x: max(x), axis = 1) # convert to 1D (remove multi-layered case for ease of analysis)
                    #     #self.opt_depth = self.opt_depth.apply(lambda x: max(x), axis = 1) # convert to 1D (remove multi-layered case for ease of analysis)
                    #     df_cirrus = pd.concat([t, pd.DataFrame(lon), pd.DataFrame(lat), cirrus_flag, self.opt_depth], axis = 1)
                    #     df_cirrus.columns = ['date', 'time', 'lon', 'lat', 'cirrus_cover', 'opt_depth']

                    #     df_cirrus = df_cirrus[subset_area].reset_index(drop=True)
                    #     df_cirrus['cirrus_cover'].at[~time_bool] = np.nan
                    #     df_cirrus['opt_depth'].at[~time_bool] = np.nan
        
                    #     cirrus_gridded = self.L2_to_L3(np.array(df_cirrus['cirrus_cover']), np.array(df_cirrus['lon']),
                    #                               np.array(df_cirrus['lat']), res_lon, res_lat, 'mean')
                    #     #self.opt_depth = self.L2_to_L3(np.array(df_cirrus['opt_depth']), np.array(df_cirrus['lon']),
                    #     #                          np.array(df_cirrus['lat']), res_lon, res_lat, 'max')
                    #     #cirrus_count = self.L2_to_L3(np.array(df_cirrus['cirrus_cover']), np.array(df_cirrus['lon']),
                    #     #                          np.array(df_cirrus['lat']), res_lon, res_lat, 'count')
                        
                    # elif self.pressure_axis == True:
                    #     self.cirrus_midlayer_pres = np.where(cirrus_flag == 1, midlayer_pres, np.nan)
                    #     self.press_layer_cirrus = np.apply_along_axis(self.bin_pressureL, 1, self.cirrus_midlayer_pres) # bin cirrus top pressure into 7 layers
    
                    #     self.cirrus_midlayer_temp = np.where((cirrus_flag == 1) &
                    #                                          (midlayer_pres >= 87.5) &
                    #                                          (midlayer_pres <= 475), midlayer_temp, np.nan)
    
                    #     # gather data and select relevant data (within ROI)
                    #     df_cirrus = pd.concat([t, pd.DataFrame(lon), pd.DataFrame(lat)], axis = 1)
                    #     df_cirrus.columns = ['date', 'time', 'lon', 'lat']
                        
                    #     subset_area = np.array([(df_cirrus['lon'] >= min_lon) & (df_cirrus['lon'] <= max_lon)
                    #                       & (df_cirrus['lat'] >= min_lat) & (df_cirrus['lat'] <= max_lat)]).reshape(-1)
                        
                    #     df_cirrus = df_cirrus[subset_area].reset_index(drop=True)
                    #     self.press_layer_cirrus = self.press_layer_cirrus[subset_area]
                    #     self.cirrus_midlayer_temp = self.cirrus_midlayer_temp[subset_area]
        
                    #     self.indices = np.where(~self.press_layer_cirrus.any(axis=1))[0] # find idx of all rows with at least 1 non-zero element
        
                    #     self.cirrus_midlayer_temp[self.indices,:] = np.nan
        
                    #     temp = self.cirrus_midlayer_temp[~np.isnan(self.cirrus_midlayer_temp)]
                        
                    #     self.temp_layer = np.empty((len(self.press_layer_cirrus),11,))
                    #     self.temp_layer.fill(np.nan)
                        
                    #     elements = np.nonzero(self.press_layer_cirrus)
          
                    #     for i, j, T in zip(elements[0], elements[1], temp):
                    #         self.temp_layer[i,j] = T
                            
                        
                    #     cirrus_gridded = []
                    #     #temp_gridded = []
                    
                    #     for col in self.press_layer_cirrus.T: # loop through pressure layers (columns)
                    #         cirrus_in_layer = pd.DataFrame(col, columns=['cirrus_cover']).reset_index(drop=True)
                    #         cirrus_in_layer['cirrus_cover'].at[~time_bool] = np.nan
                    #         df_layer = pd.concat([df_cirrus, cirrus_in_layer], axis = 1)
                    #         cirrus_gridded.append(self.L2_to_L3(np.array(df_layer['cirrus_cover']), np.array(df_cirrus['lon']),
                    #                                         np.array(df_cirrus['lat']), res_lon, res_lat, 'mean'))
                            
                    #     for col in self.temp_layer.T: # loop through pressure layers (columns)
                    #         cirrus_in_layer = pd.DataFrame(col, columns=['temperature']).reset_index(drop=True)
                    #         cirrus_in_layer['temperature'].at[~time_bool] = np.nan
                    #         df_layer = pd.concat([df_cirrus, cirrus_in_layer], axis = 1)
                    #         #temp_gridded.append(self.L2_to_L3(np.array(df_layer['temperature']), np.array(df_cirrus['lon']),
                    #         #                               np.array(df_cirrus['lat']), res_lon, res_lat, self.nanmean))
                        
                        
                    # else:
                    #     print("Input not recognized")
                    #     break
                    
                    # self.lon_pos.append(np.array(df_cirrus['lon']))
                    # self.lat_pos.append(np.array(df_cirrus['lat']))
                    
                    # mid_time = start_time_window + (end_time_window - start_time_window) / 2
                    # binned_datetime = mid_time - timedelta(seconds = mid_time.time().second)
                    
                    self.layered_cirrus.append(self.cirrus_midlayer_pres)
                    #self.CALIPSO_cirrus.append(cirrus_gridded)
                    #self.cirrus_od.append(self.opt_depth)
                    #self.CALIPSO_temp.append(temp_gridded)
                    #self.CALIPSO_count.append(cirrus_count)
                    #self.dates.append(binned_datetime.date())
                    #self.times.append(binned_datetime.time())
                    
                    start_time_window = start_time_window + timedelta(minutes = self.time_res)
                    end_time_window = start_time_window + timedelta(minutes = self.time_res)
                    
            except:
                pass
        
        #self.CALIPSO_cirrus = np.stack(self.CALIPSO_cirrus)
        #self.CALIPSO_temp = np.stack(self.CALIPSO_temp)
        #self.cirrus_od = np.stack(self.cirrus_od)
        
        #return self.CALIPSO_cirrus
    
    def positional_heatmap(self):
        self.lon_pos = np.concatenate(self.lon_pos, axis = 0)
        self.lat_pos = np.concatenate(self.lat_pos, axis = 0)
        pos_arr = np.column_stack([self.lon_pos, self.lat_pos])
        pos_df = pd.DataFrame(pos_arr, columns = ['longitude', 'latitude'])
        sns.jointplot(x = 'longitude', y = 'latitude', data = pos_df,
                      kind = 'hex', xlim = [min_lon, max_lon],
                      ylim = [min_lat, max_lat])
        plt.show()
        return self.lon_pos, np.shape(self.lon_pos)
    
    def time_map(self):

        # get all overpass times over Europe of CALIPSO
        overpass_time = [datetime.combine(date, time) for date, time in zip(self.dates, self.times)]
        overpass_time = pd.DataFrame(overpass_time, columns = ['hour of the day'])
        
        # create a frequency list of overpass times rounded down to hours 
        freqlist = overpass_time.groupby(overpass_time["hour of the day"].dt.hour).count()
        freqlist.columns = ['freq']
        
        # create df of all hours in a day (0-23), all zeros
        overpass_freq = pd.DataFrame(np.zeros((24,)), columns = ['freq'])
        
        # replace the non-zero freqs using freqlist
        overpass_freq.loc[overpass_freq.index.isin(freqlist.index), ['freq']] = freqlist[['freq']]
        overpass_freq.columns = ["nr of CALIPSO overpasses during March '15"]
        
        plt.figure(figsize = (10, 6))
        ax = plt.subplot(111, polar=True)
        
        ax.bar((overpass_freq.index / 24 + 1 / 48 ) * 2 * m.pi, 
               overpass_freq["nr of CALIPSO overpasses during March '15"], 
               width = 0.2, alpha = 0.5, color = "#f39c12")
        
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=14)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks(np.arange(0, 2 * m.pi + 2 / 24 * m.pi, 2 / 24 * m.pi))
        ax.set_yticks(np.arange(0, 25, 5))
        ax.set_title("Overpass frequency of CALIPSO over Europe in {0}-{1}".format(self.directory_name[-5:-3], self.directory_name[-2:]))
        ticks = [f"{i}:00" for i in range(0, 24, 1)]
        ax.set_xticklabels(ticks)
    
    def perform_animation(self, save): # save as boolean
        '''
        --------------------------ANIMATE CALIPSO-----------------------------
        '''
        
        fig = plt.figure(figsize=(16,12))
        ax = plt.subplot(111)
        plt.rcParams.update({'font.size': 15})#-- create map
        bmap = Basemap(projection='cyl',llcrnrlat= 35.,urcrnrlat= 60.,\
                      resolution='l',  llcrnrlon=-10.,urcrnrlon=40.)
        #-- draw coastlines and country boundaries, edge of map
        bmap.drawcoastlines()
        bmap.drawcountries()
        bmap.bluemarble()
        
        #-- create and draw meridians and parallels grid lines
        bmap.drawparallels(np.arange( -90., 90.,10.),labels=[1,0,0,0],fontsize=10)
        bmap.drawmeridians(np.arange(-180.,180.,10.),labels=[0,0,0,1],fontsize=10)
        
        lons, lats = bmap(*np.meshgrid(np.arange(-10, 40, 0.25), np.arange(35, 60, 0.25)))
        
        # contourf 
        im = bmap.contourf(lons, lats, self.cirrus_ani[0].T, np.arange(0, 1.01, 0.01), 
                           extend='neither', cmap='jet')
        cbar = plt.colorbar(im)
        title = ax.text(0.5, 1.05, self.dates[0],
                            ha="center", transform=ax.transAxes,)
        
        def animate(i):
            global im, title
            for c in im.collections:
                c.remove()
            title.remove()
            im = bmap.contourf(lons, lats, self.cirrus_ani[i].T, np.arange(0, 1.01, 0.01), 
                               extend='neither', cmap='jet')
            title = ax.text(0.5,1.05,"Cirrus cover from CALIPSO\n{0} {1}".format(self.dates[i], self.times[i]),
                            ha="center", transform=ax.transAxes,)
            
        myAnimation = animation.FuncAnimation(fig, animate, frames = len(self.cirrus_ani), interval = 1000)
        
        if save == True:
            # Set up formatting for the movie files
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
            myAnimation.save('CALIPSO.mp4', writer=writer)

#%%

'''
-----------------------------PROCESS ERA5 DATA---------------------------------
'''

# path = "E:\\Research_cirrus\\ERA5_data"

# def netcdf_import(file, *args):
#     data = Dataset(file,'r')
#     #print(data.variables)
#     try:
#         pres_level = args[0]
#     except:
#         pres_level = np.nan
#     #print(data.variables.keys())
#     my_dict = {}
#     if 'level' in data.variables.keys():
#         try:
#             idx = np.where(data['level'][:] == pres_level)[0][0]
#         except:
#             print("Define a pressure level!")
#             return
#     for key in data.variables.keys():
#         if key == 'longitude' or key == 'latitude' or key == 'time' or key == 'level':
#             my_dict[key] = data[key][:]
#         else:
#             my_dict[key] = data[key][:, idx, :, :]
#     return my_dict

# ERA5dict = netcdf_import(path + '\\ERA5_15.nc', 250)

# #def animate(dataset_dict, var):
# fig = plt.figure(figsize=(16,12))
# ax = plt.subplot(111)
# plt.rcParams.update({'font.size': 15})#-- create map
# map = Basemap(projection='cyl',llcrnrlat= 35.,urcrnrlat= 60.,\
#               resolution='l',  llcrnrlon=-10.,urcrnrlon=40.)
# #-- draw coastlines and country boundaries, edge of map
# map.drawcoastlines()
# map.drawcountries()
# map.bluemarble()

# #-- create and draw meridians and parallels grid lines
# map.drawparallels(np.arange( -90., 90.,30.),labels=[1,0,0,0],fontsize=10)
# map.drawmeridians(np.arange(-180.,180.,30.),labels=[0,0,0,1],fontsize=10)

# lons, lats = map(*np.meshgrid(ERA5dict['longitude'], ERA5dict['latitude']))

# # contourf 
# im = map.contourf(lons, lats, ERA5dict['r'][0], np.arange(0, 100.1, 0.1), extend='both', cmap='jet')
# cbar=plt.colorbar(im)
# date = '01-03-2015 00:00:00'
# time = datetime.strptime(date, '%d-%m-%Y %H:%M:%S')
# title = ax.text(0.5,1.05,time,
#                     ha="center", transform=ax.transAxes,)

# def animate(i):
#     global im, title
#     for c in im.collections:
#         c.remove()
#     title.remove()
#     im = map.contourf(lons, lats, ERA5dict['r'][i], np.arange(0, 100.1, 0.1), extend='both', cmap='jet')
#     date = '01-03-2015 00:00:00'
#     time = datetime.strptime(date, '%d-%m-%Y %H:%M:%S') + timedelta(hours = i)
#     title = ax.text(0.5,1.05,"RH from ERA5 Reanalysis Data\n{0}".format(time),
#                     ha="center", transform=ax.transAxes,)
    
# myAnimation = animation.FuncAnimation(fig, animate, frames = len(ERA5dict['time']))

#%%

'''
-----------------------PROCESS METEOSAT CLAAS 2.1 DATA-------------------------
'''

meteosat_path = 'E:\\Research_cirrus\\Meteosat_CLAAS_data\\'

class meteosat:
    
    def __init__(self):
        
        auxfilename = '{0}CM_SAF_CLAAS2_L2_AUX.nc'.format(meteosat_path)
        self.auxfile = Dataset(auxfilename,'r')

    def cover_retrieval(self, cover, opt_thick, idx_list):
        
        self.ct_sample = []
        self.cot_sample = []
        self.dates_meteo = []
        idx = 0
        
        quarter_hours = pd.date_range(start = datetime.strptime('01-03-2015 00:00', '%d-%m-%Y %H:%M'), 
                                  end = datetime.strptime('31-03-2015 23:45', '%d-%m-%Y %H:%M'),
                                  freq = '15min').to_pydatetime().tolist()
        
        for filename in os.listdir(meteosat_path):
            if filename.endswith("UD.nc"):
                if idx in idx_list:
                    print(idx, filename)
                    L2_data = Dataset(meteosat_path + str(filename),'r')
                    
                    # extract variables of interest
                    ct = L2_data['ct'][:]
                    
                    # filter out cirrus clouds
                    ct_cirrus = np.where(ct == 7, 1, 0) # all cirrus occurrences 1, the rest 0
                    
                    # coordinates
                    lat = self.auxfile['lat'][:]
                    lon = self.auxfile['lon'][:]
                    
                    if opt_thick == True:
                        cot = L2_data['cot'][:]
                        cot_cirrus = np.where(ct_cirrus[0] == 1, cot, np.nan) # all non-cirrus pixels NaN
                        cot_cirrus = np.where((cot_cirrus == -1) | (cot_cirrus == np.nan), np.nan, cot_cirrus) # make all invalid data (-1) NaN
                        self.cot_sample.append(miscellaneous.bin_2d(lon, lat, cot_cirrus[0], min_lon, max_lon,
                                                          min_lat, max_lat, res_lon, res_lat, stat = lambda x: np.nanmean(x)))
                    
                    if cover == True:
                        self.ct_sample.append(miscellaneous.bin_2d(lon, lat, ct_cirrus[0], min_lon, max_lon,
                                                          min_lat, max_lat, res_lon, res_lat, stat = 'mean'))
                        
                    self.dates_meteo.append(datetime.strptime(filename[5:17], '%Y%m%d%H%M'))

                        # print('not found')
                        # if cover == True:
                        #     self.ct_sample.append(np.full([int((max_lon - min_lon) / res_lon), 
                        #                                int((max_lat - min_lat) / res_lat)], np.nan))
                        # if opt_thick == True:
                        #     self.cot_sample.append(np.full([int((max_lon - min_lon) / res_lon), 
                        #                                int((max_lat - min_lat) / res_lat)], np.nan))
                    
                idx += 1
        
        try:
            self.ct_sample = np.stack(self.ct_sample)
        except:
            pass
        try:
            self.cot_sample = np.stack(self.cot_sample)
        except:
            pass

#%%

# #ct_sample = np.where(ct_sample == 0, np.nan, ct_sample)

# # Set up formatting for the movie files
# #Writer = animation.writers['ffmpeg']
# #writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])

# fig = plt.figure(figsize=(16,12))
# ax = plt.subplot(111)
# plt.rcParams.update({'font.size': 15})#-- create map
# map = Basemap(projection='cyl',llcrnrlat= 35 + res_lat/2, urcrnrlat= 60 - res_lat/2,\
#               resolution='l',  llcrnrlon=-10 + res_lon/2, urcrnrlon=40 - res_lon/2)
# #-- draw coastlines and country boundaries, edge of map
# map.drawcoastlines()
# map.drawcountries()
# map.bluemarble()

# #-- create and draw meridians and parallels grid lines
# map.drawparallels(np.arange( -90., 90.,15.),labels=[1,0,0,0],fontsize=10)
# map.drawmeridians(np.arange(-180.,180.,15.),labels=[0,0,0,1],fontsize=10)

# lons, lats = map(*np.meshgrid(np.arange(min_lon + res_lon/2, max_lon, res_lon),
#                               np.arange(min_lat + res_lat/2, max_lat, res_lat)))

# # contourf 
# im = map.contourf(lons, lats, ct_sample[0].T, np.arange(0, 1.01, 0.01), 
#                   extend='neither', cmap='binary')
# cbar=plt.colorbar(im)
# date = '28-02-2015 00:00:00'
# time = datetime.strptime(date, '%d-%m-%Y %H:%M:%S')
# title = ax.text(0.5,1.05,time,
#                     ha="center", transform=ax.transAxes,)

# def animate(i):
#     global im, title
#     for c in im.collections:
#         c.remove()
#     title.remove()
#     im = map.contourf(lons, lats, ct_sample[i].T, np.arange(0, 1.01, 0.01), 
#                       extend='neither', cmap='binary')
#     date = '01-03-2015 00:00:00'
#     time = datetime.strptime(date, '%d-%m-%Y %H:%M:%S') + timedelta(minutes = i * 15)
#     title = ax.text(0.5,1.05,"Cloud cover from SEVIRI Meteosat CLAAS\n{0}".format(time),
#                     ha="center", transform=ax.transAxes,)
    
# myAnimation = animation.FuncAnimation(fig, animate, frames = len(ct_sample))
# #myAnimation.save('cirruscoverCLAAS.mp4', writer=writer)




