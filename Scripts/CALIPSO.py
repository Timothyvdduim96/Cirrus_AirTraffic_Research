# -*- coding: utf-8 -*-
"""
@author: T Y van der Duim
"""

'''
------------------------------PACKAGES-----------------------------------------
'''

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import datetime
from pyhdf.SD import SD, SDC
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
--------------------------------PARAMS-----------------------------------------
'''

res_lon = 0.25 # set desired resolution in longitudinal direction in degs
res_lat = 0.25 # set desired resolution in latitudinal direction in degs
min_lon = -10 # research area min longitude
max_lon = 40 # research area max longitude
min_lat = 35 # research area min latitude
max_lat = 60 # research area max latitude
months_complete = ['03_15', '06_15', '09_15', '12_15', '03_16', '06_16', '09_16', '12_16',
                   '03_17', '06_17', '09_17', '12_17', '03_18', '06_18', '09_18', '12_18',
                   '03_19', '06_19', '09_19', '12_19', '03_20', '06_20', '09_20', '12_20']
months_1518 = months_complete[:16] # months till 2018
months_1920 = months_complete[16:] # months 2019 & 2020
dates = [month.replace('_', '-') for month in months_complete]
years = ['2015', '2016', '2017', '2018', '2019', '2020'] # years

'''
---------------------------------PATHS-----------------------------------------
'''

flight_path = 'E:\\Research_cirrus\\Flight_data\\'
lidar_path = 'E:\\Research_cirrus\\CALIPSO_data\\'
meteo_path = 'E:\\Research_cirrus\\ERA5_data\\'
meteosat_path = 'E:\\Research_cirrus\\Meteosat_CLAAS_data\\'

#%%
'''
------------------------------IMPORT HDF4--------------------------------------
'''

class hdf4_files:

    '''
        Parameters
        ----------
        file_name : name of hdf4 to import, str

        Returns
        -------
        file (import_hdf4) or specific file object (select_var)
    '''
    
    def __init__(self, file_name):
        self.file_name = file_name
        
    def import_hdf4(self):
        file = SD(self.file_name, SDC.READ)
        
        datasets_dic = file.datasets()
        
        for idx,sds in enumerate(datasets_dic.keys()):
            
            sds_obj = file.select(sds)
            
        return file
    
    def select_var(self, var):
        sds_obj = self.import_hdf4().select(var) # select variable
    
        data = sds_obj.get() # get sds data

        for key, value in sds_obj.attributes().items():

            if key == "_FillValue":
                fillvalue = value
            if key == 'add_offset':
                add_offset = value
            elif key == 'scale_factor':
                scale_factor = value
            else:
                add_offset = 0
                scale_factor = 1

        try:
            data = np.where(data == fillvalue, np.nan, data)
        except:
            data = data
            
        return data

#%%    
'''
---------------------------ANALYZE CALIPSO DATA--------------------------------
'''

class CALIPSO_analysis:

    '''
        Parameters
        ----------
        month : month + year (e.g. 03_15), str
        time_res : time resolution which will be the temporal resolution of the data at each overpass, int
        pressure_axis : indicates whether cirrus properties should be binned vertically or taken over the entire column, bool

        Returns
        -------
        CALIPSO gridded L3 products like cirrus cover, cloud temperature and optical depth (latter only for 5km product)
    '''
    
    def __init__(self, month, time_res, pressure_axis): # pressure axis is a boolean!
        self.directory_name = lidar_path + 'LIDAR_' + month
        self.load_hdf4 = hdf4_files
        self.time_res = time_res
        self.pressure_axis = pressure_axis
        # automatically run essential methods
        self.CALIPSO()

    def feature_class(self, var, start, end, class_flag):
            
        feature_type = {0: 'invalid', 1: 'clear air', 2: 'cloud', 3: 'aerosol', 4: 'stratospheric feature',
                    5: 'surface', 6: 'subsurface', 7: 'nosignal'}
            
        feature_QA = {0: 'none', 1: 'low', 2: 'medium', 3: 'high'}
        
        feature_subcloud = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 
                                7: '7'}
            
        '''
        feature_subcloud = {0: 'low overcast, transparent', 1: 'low overcast, opaque',
                            2: 'transition stratocumulus', 3: 'low, broken cumulus',
                            4: 'altocumulus (transparent)', 5: 'altostratus (opaque)',
                            6: 'cirrus (transparent)', 7: 'deep convective (opaque)'}
        '''
        
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
        
    def CALIPSO(self):
        
        self.layered_cirrus = []
        self.CALIPSO_cirrus = []
        self.cirrus_od = []
        self.CALIPSO_temp = []
        self.dates = []
        self.times = []
        self.lon_pos = []
        self.lat_pos = []
        
        for filename in os.listdir(self.directory_name):
            print(self.directory_name + "\\" + str(filename))
            try:
                test_calipso = self.load_hdf4(self.directory_name + "\\" + str(filename))
                # import variables of interest
                class_flag = test_calipso.select_var("Feature_Classification_Flags")
                t = test_calipso.select_var("Profile_UTC_Time")
                lat = test_calipso.select_var("Latitude")
                lon = test_calipso.select_var("Longitude")
                midlayer_pres = test_calipso.select_var("Midlayer_Pressure") 
                midlayer_temp = test_calipso.select_var("Midlayer_Temperature")
            except:
                continue
        
            try:
                opt_depth = test_calipso.select_var("Feature_Optical_Depth_532") # only in 5km product
            except:
                pass
                    
            feature_type_data = self.feature_class(1, -3, None, class_flag)
            feature_QA_data = self.feature_class(2, -5, -3, class_flag)
            feature_subtype_data = self.feature_class(3, -12, -9, class_flag)
            
            features = pd.DataFrame({'type': feature_type_data,
                                     'QA': feature_QA_data,
                                     'subtype': feature_subtype_data}, index = np.unique(class_flag))
            
            # mark cirrus with high or middle confidence 1, no cirrus 0 and cirrus with low confidence -1
            
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
            t['time'] = t['time'].apply(lambda x: miscellaneous.convert_seconds(x))
            combined_datetime = [datetime.datetime.combine(date, time) for date, time in zip(t['date'], t['time'])]
            combined_datetime = list(np.array(combined_datetime)[subset_area])
            
            if len(combined_datetime) == 0: # if there is no data within the ROI, go to next file
                continue

            start_time_window = combined_datetime[0]
            end_time_window = start_time_window + datetime.timedelta(minutes = self.time_res)
            
            while start_time_window < combined_datetime[-1]:
            
                time_bool = pd.DataFrame(combined_datetime)[0].between(start_time_window, end_time_window)
                cirrus_flag = pd.DataFrame(class_flag).replace(feature_dict) # map this coding to classification flag array
                self.cirrus_midlayer_pres = np.where(cirrus_flag == 1, midlayer_pres, np.nan)
                try:
                    opt_depth = pd.DataFrame(np.where(cirrus_flag == 1, opt_depth, 0))
                except:
                    pass

                if self.pressure_axis == False:
                    
                    cirrus_flag = cirrus_flag.apply(lambda x: max(x), axis = 1) # convert to 1D (remove multi-layered case for ease of analysis)
                    try:
                        opt_depth = opt_depth.apply(lambda x: max(x), axis = 1) # convert to 1D (remove multi-layered case for ease of analysis)
                        df_cirrus = pd.concat([t, pd.DataFrame(lon), pd.DataFrame(lat), cirrus_flag, opt_depth], axis = 1)
                        df_cirrus.columns = ['date', 'time', 'lon', 'lat', 'cirrus_cover', 'opt_depth']
                        df_cirrus = df_cirrus[subset_area].reset_index(drop=True)
                        df_cirrus['opt_depth'].at[~time_bool] = np.nan
                        self.opt_depth = miscellaneous.bin_2d(np.array(df_cirrus['lon']),
                        np.array(df_cirrus['lat']), np.array(df_cirrus['opt_depth']),
                                min_lon, max_lon, min_lat, max_lat, res_lon, res_lat, 'max')
                    except:
                        df_cirrus = pd.concat([t, pd.DataFrame(lon), pd.DataFrame(lat), cirrus_flag], axis = 1)
                        df_cirrus.columns = ['date', 'time', 'lon', 'lat', 'cirrus_cover']
                        df_cirrus = df_cirrus[subset_area].reset_index(drop=True)

                    df_cirrus['cirrus_cover'].at[~time_bool] = np.nan
                    cirrus_gridded = miscellaneous.bin_2d(np.array(df_cirrus['lon']), np.array(df_cirrus['lat']),
                                                   np.array(df_cirrus['cirrus_cover']), min_lon, max_lon,
                                                   min_lat, max_lat, res_lon, res_lat, 'mean')
                    
                elif self.pressure_axis == True:
                    self.cirrus_midlayer_pres = np.where(cirrus_flag == 1, midlayer_pres, np.nan)
                    bins = np.array([87.5, 112.5, 137.5, 162.5, 187.5, 212.5,
                                          237.5, 275, 325, 375, 425, 475])
                    
                    self.layer_cirrus = np.digitize(self.cirrus_midlayer_pres, bins)
                    
                    self.press_layer_cirrus = []
                    
                    for layer in range(11):
                        binary_lvl = np.where(self.layer_cirrus == layer + 1, 1, 0)
                        self.binary_layer_cirrus = np.apply_along_axis(sum, 1, np.array(binary_lvl)) # bin cirrus top pressure into 7 layers
                        self.press_layer_cirrus.append(self.binary_layer_cirrus)
                    
                    self.press_layer_cirrus = np.column_stack(self.press_layer_cirrus)
                    
                    self.cirrus_midlayer_temp = np.where((cirrus_flag == 1) &
                                                          (midlayer_pres >= min(bins)) &
                                                          (midlayer_pres <= max(bins)), midlayer_temp, np.nan)

                    # gather data and select relevant data (within ROI)
                    df_cirrus = pd.concat([t, pd.DataFrame(lon), pd.DataFrame(lat)], axis = 1)
                    df_cirrus.columns = ['date', 'time', 'lon', 'lat']
                    
                    subset_area = np.array([(df_cirrus['lon'] >= min_lon) & (df_cirrus['lon'] <= max_lon)
                                      & (df_cirrus['lat'] >= min_lat) & (df_cirrus['lat'] <= max_lat)]).reshape(-1)
                    
                    df_cirrus = df_cirrus[subset_area].reset_index(drop=True)
                    
                    self.press_layer_cirrus = self.press_layer_cirrus[subset_area]
                    self.cirrus_midlayer_temp = self.cirrus_midlayer_temp[subset_area]
    
                    self.indices = np.where(~self.press_layer_cirrus.any(axis=1))[0] # find idx of all rows with at least 1 non-zero element
    
                    self.cirrus_midlayer_temp[self.indices,:] = np.nan
    
                    temp = self.cirrus_midlayer_temp[~np.isnan(self.cirrus_midlayer_temp)]
                    
                    self.temp_layer = np.empty((len(self.press_layer_cirrus), len(bins) - 1,))
                    self.temp_layer.fill(np.nan)
                    
                    elements = np.nonzero(self.press_layer_cirrus)
      
                    for i, j, T in zip(elements[0], elements[1], temp):
                        self.temp_layer[i,j] = T
                    
                    cirrus_gridded = []
                    temp_gridded = []
                
                    for col in self.press_layer_cirrus.T: # loop through pressure layers (columns)
                        cirrus_in_layer = pd.DataFrame(col, columns=['cirrus_cover']).reset_index(drop=True)
                        cirrus_in_layer['cirrus_cover'].at[~time_bool] = np.nan
                        df_layer = pd.concat([df_cirrus, cirrus_in_layer], axis = 1)
                        cirrus_gridded.append(miscellaneous.bin_2d(np.array(df_cirrus['lon']),
                                                        np.array(df_cirrus['lat']), np.array(df_layer['cirrus_cover']), min_lon,
                                                        max_lon, min_lat, max_lat, res_lon, res_lat, 'mean'))
                        
                    for col in self.temp_layer.T: # loop through pressure layers (columns)
                        cirrus_in_layer = pd.DataFrame(col, columns=['temperature']).reset_index(drop=True)
                        cirrus_in_layer['temperature'].at[~time_bool] = np.nan
                        df_layer = pd.concat([df_cirrus, cirrus_in_layer], axis = 1)
                        temp_gridded.append(miscellaneous.bin_2d(np.array(df_cirrus['lon']),
                                                        np.array(df_cirrus['lat']), np.array(df_layer['temperature']), min_lon,
                                                        max_lon, min_lat, max_lat, res_lon, res_lat, lambda x: np.nanmean(x)))
                    
                    
                else:
                    print("Input not recognized")
                    break
                
                mid_time = start_time_window + (end_time_window - start_time_window) / 2
                binned_datetime = mid_time - datetime.timedelta(seconds = mid_time.time().second)
                
                # append to list
                self.layered_cirrus.append(self.cirrus_midlayer_pres)
                self.CALIPSO_cirrus.append(cirrus_gridded)
                try:
                    self.cirrus_od.append(self.opt_depth)
                except:
                    pass
                try:
                    self.CALIPSO_temp.append(temp_gridded)
                except:
                    pass
                self.dates.append(binned_datetime.date())
                self.times.append(binned_datetime.time())
                self.lon_pos.append(np.array(df_cirrus['lon']))
                self.lat_pos.append(np.array(df_cirrus['lat']))
                
                start_time_window +=  datetime.timedelta(minutes = self.time_res)
                end_time_window +=  datetime.timedelta(minutes = self.time_res)
        
        self.CALIPSO_cirrus = np.stack(self.CALIPSO_cirrus)
        
        try:
            self.CALIPSO_temp = np.stack(self.CALIPSO_temp)
        except:
            pass
        
        try:
            self.cirrus_od = np.stack(self.cirrus_od)
        except:
            pass
        
        return self.CALIPSO_cirrus

#%%
'''
---------------------------VISUALS CALIPSO DATA--------------------------------
'''

class CALIPSO_visuals:
    
    '''
        possibility to plot a positional heatmap or time map for any month
    '''
    
    def positional_heatmap(self, lon_pos, lat_pos):
        self.lon_pos = np.concatenate(lon_pos, axis = 0)
        self.lat_pos = np.concatenate(lat_pos, axis = 0)
        pos_arr = np.column_stack([self.lon_pos, self.lat_pos])
        pos_df = pd.DataFrame(pos_arr, columns = ['longitude', 'latitude'])
        sns.jointplot(x = 'longitude', y = 'latitude', data = pos_df,
                      kind = 'hex', xlim = [min_lon, max_lon],
                      ylim = [min_lat, max_lat])
        plt.show()
    
    def time_map(dates, times):

        # get all overpass times over Europe of CALIPSO
        overpass_time = [datetime.datetime.combine(date, time) for date, time in zip(dates, times)]
        overpass_time = pd.DataFrame(overpass_time, columns = ['hour of the day'])
        
        # create a frequency list of overpass times rounded down to hours 
        freqlist = overpass_time.groupby(overpass_time["hour of the day"].dt.hour).count()
        freqlist.columns = ['freq']
        
        # create df of all hours in a day (0-23), all zeros
        overpass_freq = pd.DataFrame(np.zeros((24,)), columns = ['freq'])
        
        # replace the non-zero freqs using freqlist
        overpass_freq.loc[overpass_freq.index.isin(freqlist.index), ['freq']] = freqlist[['freq']]
        overpass_freq.columns = ["nr of CALIPSO overpasses"]
        
        plt.figure(figsize = (10, 6))
        ax = plt.subplot(111, polar=True)
        ax.bar((overpass_freq.index / 24 + 1 / 48 ) * 2 * m.pi, 
               overpass_freq["nr of CALIPSO overpasses"], 
               width = 0.2, alpha = 0.5, color = "green")
        
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=14)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks(np.arange(0, 2 * m.pi + 2 / 24 * m.pi, 2 / 24 * m.pi))
        #ax.set_yticks(np.arange(0, 550, 100))
        ax.set_rlabel_position(83.5)
        ticks = [f"{i}:00" for i in range(0, 24, 1)]
        ax.set_xticklabels(ticks)
        
    def exe_CALIPSO_visuals(self):
        
        '''
        run this method to get geographic positions and times of passage for all months in analysis
        '''
        
        lon_pos = []
        lat_pos = []
        alldates = []
        alltimes = []
        
        for month in months_complete:
            calipso = CALIPSO_analysis(month, 15, False)
            lon_pos.append(calipso.lon_pos)
            lat_pos.append(calipso.lat_pos)
            alldates.append(calipso.dates)
            alltimes.append(calipso.times)
        
        lon_pos = [np.concatenate(ps, axis = 0) for ps in lon_pos]
        self.lon_pos = np.concatenate(lon_pos, axis = 0)
        lat_pos = [np.concatenate(ps, axis = 0) for ps in lat_pos]
        self.lat_pos = np.concatenate(lat_pos, axis = 0)
        self.all_dates = np.concatenate(alldates, axis = 0)
        self.all_times = np.concatenate(alltimes, axis = 0)
        


