# -*- coding: utf-8 -*-
"""
@author: T Y van der Duim
"""

'''
------------------------------PACKAGES-----------------------------------------
'''

import numpy as np
from miscellaneous import miscellaneous
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from netCDF4 import Dataset
import datetime
import matplotlib.dates as mdates
import scipy.stats as st
import jenkspy
import os
from AIRCRAFT import flight_analysis

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
----------------PROCESS METEOSAT CLAAS 2.1 DATA MAR 2015-----------------------
'''

class meteosat:
    
    '''
        Parameters
        ----------
        get_ct : extract L3 cirrus cover, boolean
        get_cot : extract L3 cirrus opt. dep., boolean
        idx_list : list of indices of to be extracted files, list

        Returns
        -------
        file (import_hdf4) or specific file object (select_var)
    '''
    
    def __init__(self, get_ct, get_cot, idx_list):
        
        auxfilename = '{0}CM_SAF_CLAAS2_L2_AUX.nc'.format(meteosat_path)
        self.auxfile = Dataset(auxfilename,'r')
        # coordinates
        self.lat = self.auxfile['lat'][:]
        self.lon = self.auxfile['lon'][:]
        self.get_ct = get_ct
        self.get_cot = get_cot
        self.idx_list = idx_list
        self.meteosat_retrieval()

    def meteosat_retrieval(self):
        
        self.ct_sample = []
        self.ctp_grid = []
        self.cot_sample = []
        self.dates_meteo = []
        
        idx = 0
        
        for filename in os.listdir(meteosat_path):
            if filename.endswith("UD.nc"):
                if idx in self.idx_list:
                    print(idx, filename)
                    L2_data = Dataset(meteosat_path + str(filename), 'r')
                    L2_data_pressure = Dataset(meteosat_path + 'height\\' + 'CTX'+ str(filename)[3:], 'r')
                    
                    if self.get_cot == True or self.get_ct == True:
                        
                        ct = L2_data['ct'][:] # get cloud type
                        ct_cirrus = np.where(ct == 7, 1, 0) # all cirrus occurrences 1, the rest 0
                    
                    if self.get_cot == True:
                        cot = L2_data['cot'][:]
                        cot_cirrus = np.where(ct_cirrus[0] == 1, cot, np.nan) # all non-cirrus pixels NaN
                        cot_cirrus = np.where((cot_cirrus == -1) | (cot_cirrus == np.nan), np.nan, cot_cirrus) # make all invalid data (-1) NaN
                        self.cot_sample.append(miscellaneous.bin_2d(self.lon, self.lat, cot_cirrus[0], min_lon, max_lon,
                                                          min_lat, max_lat, res_lon, res_lat, stat = lambda x: np.nanmean(x)))
                    
                    if self.get_ct == True:
                        ctp = L2_data_pressure['ctp'][:] # get cloud top pressure
                        ctp = np.where(ct_cirrus[0] == 1, ctp, np.nan)
        
                        self.ct_sample.append(miscellaneous.bin_2d(self.lon, self.lat, ct_cirrus[0], min_lon, max_lon,
                                                          min_lat, max_lat, res_lon, res_lat, stat = 'mean'))
                        self.ctp_grid.append(miscellaneous.bin_2d(self.lon, self.lat, ctp[0], min_lon, max_lon,
                                                          min_lat, max_lat, res_lon, res_lat, stat = 'mean'))
                        
                    self.dates_meteo.append(datetime.datetime.strptime(filename[5:17], '%Y%m%d%H%M'))
                    
                idx += 1
        
        try:
            self.ct_sample = np.stack(self.ct_sample)
        except:
            pass
        
        try:
            self.ctp_grid = np.stack(self.ctp_grid)
        except:
            pass
        
        try:
            self.cot_sample = np.stack(self.cot_sample)
        except:
            pass

#%%

class METEOSAT_analysis:
        
    def __init__(self):
        
        self.quarter_hours = pd.date_range(start = datetime.datetime.strptime('01-03-2015 00:00', '%d-%m-%Y %H:%M'), 
                                  end = datetime.datetime.strptime('31-03-2015 23:45', '%d-%m-%Y %H:%M'),
                                  freq = '15min').to_pydatetime().tolist()
        self.dim_lon = int((max_lon - min_lon) / res_lon)
        self.dim_lat = int((max_lat - min_lat) / res_lat)
        
    def exe_meteo(self):
        self.cirrus_dynamics()
        self.flights_Mar15()
        self.meteo_arrays()
        
    def cirrus_dynamics(self):

        self.quarter_hours_day = pd.DataFrame(np.arange(0, len(self.quarter_hours)), 
                                              index = self.quarter_hours, 
                                              columns = ['index'])#.between_time('07:00', '18:00')
        
        index = self.quarter_hours_day['index'].tolist()
        
        self.quarter_hours_day = [self.quarter_hours[idx] for idx in index]
        
        # retrieve meteosat data
        init_meteosat = meteosat(get_ct = True, get_cot = False, 
                                                    idx_list = index)
        
        self.cirrus_cover = init_meteosat.ct_sample
        self.ctp_grid = init_meteosat.ctp_grid
        self.dates_meteo = init_meteosat.dates_meteo
        
        if len(self.quarter_hours_day) != len(self.dates_meteo):
            missing_times = set(self.quarter_hours_day).difference(self.dates_meteo)
            indices_missing = [self.quarter_hours_day.index(x) for x in missing_times]
            nanarray = np.full([int((max_lon - min_lon) / res_lon), int((max_lat - min_lat) / res_lat)], np.nan)
            
            self.cirrus_cover = np.insert(self.cirrus_cover, indices_missing, nanarray, axis = 0)
            self.ctp_grid = np.insert(self.ctp_grid, indices_missing, nanarray, axis = 0)
        
        self.diffs = np.diff(self.cirrus_cover, axis = 0)
                
    def flights_Mar15(self):
        
        self.quarter_hours = pd.date_range(start = datetime.datetime.strptime('01-03-2015 00:00', '%d-%m-%Y %H:%M'), 
                              end = datetime.datetime.strptime('31-03-2015 23:45', '%d-%m-%Y %H:%M'),
                              freq = '15min').to_pydatetime().tolist()
        self.quarter_hours_day = pd.DataFrame(np.arange(0, len(self.quarter_hours)), 
                                          index = self.quarter_hours, 
                                          columns = ['index'])#.between_time('07:00', '18:00')
        index = self.quarter_hours_day['index'].tolist()
    
        self.quarter_hours_day = [self.quarter_hours[idx] for idx in index]
        self.flights = flight_analysis('03_15')
        self.flights.individual_flights()
        self.flights.flighttracks()
        self.flights.merge_datasets()
        self.flights.grid_ATD(self.flights.flights_merged, [6, 14], self.quarter_hours_day, 15)
        self.flights_ATD = self.flights.flights_ATD
        self.flights_ATD = self.flights_ATD[:-1, :, :]
                    
    def meteo_arrays(self):
        
        bins = np.array([87.5, 112.5, 137.5, 162.5, 187.5, 212.5,
                                          237.5, 275, 325, 375, 425, 475])
        
        ctp_grid = np.digitize(self.ctp_grid, bins)
        
        ctp_grid = np.where(ctp_grid == len(bins), np.nan, ctp_grid)
        
        ERA_Data = miscellaneous.netcdf_import(meteo_path + 'NewERA_5\\' + 'ERA5_' + '03_15' + '.nc')
        ERA_temp = ERA_Data['t']
        ERA_relhum = ERA_Data['r']
        ERA_relhum = np.transpose(ERA_relhum, (0, 1, 3, 2))[:, :, :-1, :-1] 
        ERA_temp = np.transpose(ERA_temp, (0, 1, 3, 2))[:, :, :-1, :-1]
        
        index_pres = ctp_grid
        
        ERA_relhum = np.reshape(ERA_relhum, (-1, 200, 100))
        ERA_temp = np.reshape(ERA_temp, (-1, 200, 100))
        
        lon = np.arange(-10, 40, 0.25)
        lat = np.arange(35, 60, 0.25)
        lon, lat = np.meshgrid(lon, lat)
        
        ERA_RH_agg = []
        ERA_temp_agg = []
                
        for i in range(len(ERA_relhum)):
            print(i)
            ERA_RH_agg.append(miscellaneous.bin_2d(lon, lat, ERA_relhum[i], min_lon, max_lon,
                                                          min_lat, max_lat, res_lon, res_lat, stat = 'mean'))
            ERA_temp_agg.append(miscellaneous.bin_2d(lon, lat, ERA_temp[i], min_lon, max_lon,
                                                          min_lat, max_lat, res_lon, res_lat, stat = 'mean'))
        
        ERA_RH_agg = np.stack(ERA_RH_agg)
        ERA_temp_agg = np.stack(ERA_temp_agg)
        
        ERA_relhum_vertmax = []
        ERA_temp_vertmax = []
        
        for i in range(0, len(ERA_temp_agg), 11):
            print(i)
            full_data_RH = ERA_RH_agg[i:int(i + 11)]
            full_data_temp = ERA_temp_agg[i:int(i + 11)]
            ERA_relhum_vertmax.append(miscellaneous.multidim_slice(index_pres[int(i / 11)], full_data_RH))
            ERA_temp_vertmax.append(miscellaneous.multidim_slice(index_pres[int(i / 11)], full_data_temp))
                
        ERA_relhum = np.stack(ERA_relhum_vertmax)
        ERA_temp = np.stack(ERA_temp_vertmax)
        
        # meteo per 15mins instead of hours (repeat 4x)
        ERA_temp = np.repeat(ERA_temp, 4, axis = 0)
        ERA_relhum = np.repeat(ERA_relhum, 4, axis = 0)
        
        self.ERA_temp = ERA_temp[:-1, :, :]
        self.ERA_relhum = ERA_relhum[:-1, :, :]
    
    def bin_atd_meteo(self):
        
        self.df = pd.DataFrame(list(zip(np.reshape(self.flights_ATD, -1), 
                            np.reshape(self.diffs, -1), np.reshape(self.ERA_temp, -1),
                            np.reshape(self.ERA_relhum, -1))), columns = ['ATD', 'delta_cirrus', 'temp', 'RH'])

        break_1 = []
        break_2 = []
        break_3 = []
        break_4 = []
        maxes = []
                
        for hr in range(96):
            print(hr)
            breaks = jenkspy.jenks_breaks(np.reshape(self.flights_ATD[hr, :, :], -1), nb_class = 5)
            break_1.append(breaks[1])
            break_2.append(breaks[2] - breaks[1])
            break_3.append(breaks[3] - breaks[2])
            break_4.append(breaks[4] - breaks[3])
            maxes.append(np.max(np.reshape(self.flights_ATD[hr, :, :], -1)) - breaks[4])
            
        day_quarter_hours = pd.date_range(start = datetime.datetime.strptime('01-03-2015 00:00', '%d-%m-%Y %H:%M'), 
                                                  end = datetime.datetime.strptime('01-03-2015 23:45', '%d-%m-%Y %H:%M'),
                                                  freq = '15min').to_pydatetime().tolist()
        
        day_quarter_hours = [t.strftime('%H:%M') for t in day_quarter_hours]
        
        break_points = pd.DataFrame(list(zip(break_1, break_2, break_3, break_4, maxes)), 
                                    index = day_quarter_hours,
                                    columns = ['very low ATD', 'low ATD', 'moderate ATD',
                                                'high ATD', 'very high ATD'])
        
        break_points.loc['mean'] = break_points.mean()

        fig, ax = plt.subplots()
        plt.rc('xtick', labelsize=14) 
        plt.rc('ytick', labelsize=20)
        break_points.plot(kind = 'bar', stacked=True, ax = ax)
        plt.ylabel('ATD (m km$^{-2}$ hr$^{-1}$)')
        ax.xaxis.set_tick_params(rotation=30)
        ticks = ax.xaxis.get_ticklocs()
        ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
        ax.xaxis.set_ticks(ticks[::4])
        ax.xaxis.set_ticklabels(ticklabels[::4])
        legend = plt.legend(frameon = 1)
        frame = legend.get_frame()
        frame.set_color('white')
        
        breaks = break_points.loc['mean'].tolist()
        breaks.insert(0, 0)
        breaks[-1] = 1e3
        print(breaks)
        self.df['cut_jenks'] = pd.cut(self.df['ATD'],
                                bins = breaks,
                                labels = ['very low ATD', 'low ATD', 'moderate ATD', 
                                          'high ATD', 'very high ATD'],
                                include_lowest = True)
                
    def saturation_split(self):
        
        # supersaturated air
        df_supersat = self.df[(self.df['RH'] >= 100) & (self.df['temp'] <= 220)]
        
        self.cut_counts_supersat = df_supersat['cut_jenks'].value_counts()
        
        self.df_grouped_supersat = df_supersat.groupby(by = 'cut_jenks').mean()
        
        self.slope_supersat, self.std_err_supersat, self.p_val_supersat = miscellaneous.coeff_linregr(self.df_grouped_supersat['delta_cirrus'].tolist(), 
                                              self.df_grouped_supersat['ATD'].tolist(), 5)
        
        # supersaturated air
        df_subsat = self.df[(self.df['RH'] <= 100)]
        
        self.cut_counts_subsat = df_subsat['cut_jenks'].value_counts()
        
        self.df_grouped_subsat = df_subsat.groupby(by = 'cut_jenks').mean()
        
        self.slope_subsat, self.std_err_subsat, self.p_val_subsat = miscellaneous.coeff_linregr(self.df_grouped_subsat['delta_cirrus'].tolist(), 
                                              self.df_grouped_subsat['ATD'].tolist(), 5)
        
        df_supersat['saturation'] = 'supersaturated'
        df_subsat['saturation'] = 'subsaturated'
        
        self.df_total = pd.concat([df_supersat, df_subsat])
        self.df_total['delta_cirrus'] = self.df_total['delta_cirrus'] * 1000 # convert delta cirrus to permilles

        
    def validation(self):
        
        lidar = np.load(lidar_path + 'LIDAR_processed\\LIDAR_' + months_complete[0] + '_5km' + '.npy', 
                                        allow_pickle = True)
        
        self.calipso = lidar.item()['calipso']
        self.cirrus_od = lidar.item()['opt_depth']
        self.dates = lidar.item()['dates']
        self.times = lidar.item()['times']
        
        self.overpass_time = [miscellaneous.quarterhr_rounder(datetime.datetime.combine(date, time)) for date, time in 
                    zip(self.dates, self.times)]
        
        self.daypasses = pd.DataFrame(np.arange(0, len(self.overpass_time)), index = self.overpass_time,
                     columns = ['index']).between_time('10:00', '15:00')
        
        index = self.daypasses['index'].tolist()
        
        self.overpass_time = [self.overpass_time[idx] for idx in index]
        self.calipso = [self.calipso[idx] for idx in index]
        self.cirrus_od = [self.cirrus_od[idx] for idx in index]
        
        self.flight_over_moment = np.where([any(x == time for x in self.overpass_time) for 
                                            time in self.quarter_hours])[0]
        
        init_meteosat = meteosat(get_ct = True, get_cot = False, 
                                                    idx_list = list(self.flight_over_moment))
        
        self.meteosat_ct = init_meteosat.ct_sample
        self.meteosat_ct = np.where(np.isnan(self.calipso) == True, np.nan, self.meteosat_ct)
        self.meteosat_cot = init_meteosat.cot_sample
        self.meteosat_cot = np.where(np.isnan(self.calipso) == True, np.nan, self.meteosat_cot)
        
        self.cirrus_od = np.where(np.isnan(self.calipso) == True, np.nan, self.cirrus_od)
        self.meteosat_ct = miscellaneous.flatten_clean(self.meteosat_ct)
        self.meteosat_cot = miscellaneous.flatten_clean(self.meteosat_cot)
        self.calipso = miscellaneous.flatten_clean(self.calipso)
        self.cirrus_od = miscellaneous.flatten_clean(self.cirrus_od)
        
        self.crosstab = pd.crosstab(np.where(self.meteosat_ct > 0, 1, 0), 
                                    np.where(self.calipso > 0, 1, 0), margins = True)
        self.crosstab.index = ['No cirrus (METEOSAT)', 'Cirrus (METEOSAT)', 'Total']
        self.crosstab.columns = ['No cirrus (CALIPSO)', 'Cirrus (CALIPSO)', 'Total']
    
    def validation_plot(self):
        
        CAL_yes_METEO_no = np.where((self.meteosat_ct == 0) & (self.calipso > 0))[0]
        CAL_yes_METEO_yes = np.where((self.meteosat_ct > 0) & (self.calipso > 0))[0]
        CAL_yes_METEO_no_od = self.cirrus_od[CAL_yes_METEO_no]
        CAL_yes_METEO_yes_od = self.cirrus_od[CAL_yes_METEO_yes]
        
        my_dict = dict({'CAL:1, METEOSAT:0': CAL_yes_METEO_no_od, 'CAL:1, METEOSAT:1': CAL_yes_METEO_yes_od})
        od_df = pd.DataFrame.from_dict(my_dict, orient = 'index')
        od_df = od_df.transpose()
        
        plt.figure(figsize = (10,10))
        od_df['CAL:1, METEOSAT:0'].hist(density = 1, histtype = 'stepfilled', alpha = .5, 
                                        bins = 100, label = 'CALIPSO')
        od_df['CAL:1, METEOSAT:1'].hist(density = 1, histtype = 'stepfilled', alpha = .5, 
                                        color = sns.desaturate("indianred", .75),
                                        bins = 100, label = 'CALIPSO & Meteosat')
        legend = plt.legend(title = 'Cirrus detected by:', frameon = 1)
        frame = legend.get_frame()
        frame.set_color('white')
        plt.xlabel('COT cirrus')
        plt.xlim([0, 2])
        plt.ylabel('norm. prob. density')
        
    '''
    -----------------------------VISUALS--------------------------------------
    '''
    
    def pointplot(self):
        
        plt.figure()
        plt.rc('xtick', labelsize=20) 
        plt.rc('ytick', labelsize=20)
        sns.pointplot(x = 'cut_jenks', y = 'delta_cirrus', hue='saturation', data = self.df_total,
                      capsize=.2, dodge = True)
        plt.ylabel('$\Delta$cirrus ({0})'.format('\u2030'))
        plt.xlabel(' ')
        
    def boxplot_ATD(self):
        
        plt.figure()
        sns.boxplot(x = 'cut_jenks', y = 'ATD', data = self.df_total)
        plt.ylabel('Air Traffic Density (m km$^{-2}$ hr$^{-1}$)')
        
    def ERA5_plot(self):
        
        hours = pd.date_range(start = datetime.datetime.strptime('01-03-2015 00:00', '%d-%m-%Y %H:%M'), 
                                  end = datetime.datetime.strptime('31-03-2015 23:00', '%d-%m-%Y %H:%M'),
                                  freq = 'H').to_pydatetime().tolist()

        ERA_Data = miscellaneous.netcdf_import(meteo_path + 'ERA5_' + '03_15' + '.nc')
        ERA_temp = ERA_Data['t']
        ERA_relhum = ERA_Data['r']
        ERA_temp = np.mean(ERA_temp, axis = (1, 2, 3))
        ERA_relhum = np.mean(ERA_relhum, axis = (1, 2, 3))
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(hours, ERA_temp, color = 'r', label = 'temperature')
        ax2.plot(hours, ERA_relhum, color = 'b', label = 'RH')
        
        locator = mdates.AutoDateLocator(minticks=31, maxticks=31)
        formatter = mdates.ConciseDateFormatter(locator)
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)
        
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        legend = ax2.legend(lines + lines2, labels + labels2, loc=9, frameon = 1)
        frame = legend.get_frame()
        frame.set_color('white')
        ax2.grid(False)
        ax1.set_xlabel('date')
        ax1.set_ylabel('temperature (K)')
        ax2.set_ylabel('RH (%)')

    def ATD_vs_CIRR(self):
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(self.quarter_hours_day, np.mean(self.flights_ATD, axis = (1, 2)), color = 'b', 
                 label = 'mean ATD')
        ax2.plot(self.quarter_hours_day, np.mean(self.cirrus_cover, (1,2)), color = 'r', 
                 label = 'mean cirrus cover')
        
        locator = mdates.AutoDateLocator(minticks=31, maxticks=31)
        formatter = mdates.ConciseDateFormatter(locator)
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)
            
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)
        ax2.grid(False)
        ax1.set_xlabel('date')
        ax1.set_ylabel('ATD (m km$^{-2}$ hr$^{-1}$)')
        ax2.set_ylabel('cirrus cover')

    def daily_cycle(self):
        
        day_quarter_hours = pd.date_range(start = datetime.datetime.strptime('01-03-2015 00:00', '%d-%m-%Y %H:%M'), 
                                          end = datetime.datetime.strptime('01-03-2015 23:45', '%d-%m-%Y %H:%M'),
                                          freq = '15min').to_pydatetime().tolist()

        flights_ATD = np.reshape(self.flights_ATD, (31, -1, self.dim_lon, self.dim_lat))
        flights_ATD = np.mean(flights_ATD, axis = (2, 3))
        cirrus_cover = np.reshape(self.cirrus_cover, (31, -1, self.dim_lon, self.dim_lat))
        cirrus_cover = np.mean(cirrus_cover, axis = (2, 3))
        
        mean_flights_ATD = np.mean(flights_ATD, axis = 0)
        mean_cirrus_cover = np.nanmean(cirrus_cover, axis = 0)
        
        conf_int = lambda x: st.t.interval(alpha=0.95, df=len(x)-1, loc=np.nanmean(x), scale=st.sem(x, nan_policy='omit'))
        
        ci_ATD_lower, ci_ATD_upper = zip(*[conf_int(row) for row in flights_ATD.T])
        ci_cirrus_lower, ci_cirrus_upper = zip(*[conf_int(row) for row in cirrus_cover.T])
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(day_quarter_hours, mean_flights_ATD, color = 'b', label = 'daily mean ATD')
        ax1.fill_between(day_quarter_hours, ci_ATD_lower, 
                         ci_ATD_upper, color='b', alpha=.1)
        ax2.plot(day_quarter_hours, mean_cirrus_cover, color = 'r', label = 'daily mean cirrus cover')
        ax2.fill_between(day_quarter_hours, ci_cirrus_lower, 
                         ci_cirrus_upper, color='r', alpha=.1)
        myFmt = mdates.DateFormatter('%H:%M')
        ax1.xaxis.set_major_formatter(myFmt)
        
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)
        ax2.grid(False)
        ax1.set_xlabel('time')
        ax1.set_ylabel('ATD (m km$^{-2}$ hr$^{-1}$)')
        ax2.set_ylabel('cirrus cover')
        
    def ISSR(self):
        
        hours = pd.date_range(start = datetime.datetime.strptime('01-03-2015 00:00', '%d-%m-%Y %H:%M'), 
                                  end = datetime.datetime.strptime('31-03-2015 23:00', '%d-%m-%Y %H:%M'),
                                  freq = 'H').to_pydatetime().tolist()

        ERA_Data = miscellaneous.netcdf_import(meteo_path + 'ERA5_' + '03_15' + '.nc')
        ERA_temp = ERA_Data['t']
        ERA_relhum = ERA_Data['r']
        
        ERA_relhum = np.transpose(ERA_relhum, (0, 1, 3, 2))[:, :, :-1, :-1] 
        ERA_temp = np.transpose(ERA_temp, (0, 1, 3, 2))[:, :, :-1, :-1]
        
        ISSR = []
        
        for idx in range(np.shape(ERA_relhum)[0]):
            ISSR.append(np.count_nonzero(ERA_relhum[idx, :, :, :] > 100) / ERA_relhum[idx, :, :, :].size * 100)
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(hours, ISSR, color = 'g', label = 'ISSR (%)')
        ax2.plot(self.quarter_hours_day, np.mean(self.cirrus_cover, (1,2)), color = 'r', 
                  label = 'mean cirrus cover', alpha = 0.2)
        
        locator = mdates.AutoDateLocator(minticks=31, maxticks=31)
        formatter = mdates.ConciseDateFormatter(locator)
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)
            
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)
        ax2.grid(False)
        ax1.set_xlabel('date')
        ax1.set_ylabel('ISSR (%)')
        ax2.set_ylabel('cirrus cover')
        
    def ISSR_daily_cycle(self):
        
        hours = pd.date_range(start = datetime.datetime.strptime('01-03-2015 00:00', '%d-%m-%Y %H:%M'), 
                                  end = datetime.datetime.strptime('01-03-2015 23:00', '%d-%m-%Y %H:%M'),
                                  freq = 'H').to_pydatetime().tolist()

        ERA_Data = miscellaneous.netcdf_import(meteo_path + 'ERA5_' + '03_15' + '.nc')
        ERA_relhum = ERA_Data['r']
        
        ERA_relhum = np.transpose(ERA_relhum, (0, 1, 3, 2))[:, :, :-1, :-1] 
        
        ERA_relhum = np.reshape(ERA_relhum, (31, -1, 7, self.dim_lon, self.dim_lat))
        issr = []
        
        for idx in range(np.shape(ERA_relhum)[1]):
            iss = []
            for idy in range(np.shape(ERA_relhum)[0]):
                iss.append(np.count_nonzero(ERA_relhum[idy, idx, :, :] >= 100) / ERA_relhum[idy, idx, :, :].size * 100)
            issr.append(iss)
                
        issr_mean = np.mean(np.array(issr), axis = 1)
        
        conf_int = lambda x: st.t.interval(alpha=0.95, df=len(x)-1, loc=np.nanmean(x), scale=st.sem(x, nan_policy='omit'))
        
        ci_lower, ci_upper = zip(*[conf_int(row) for row in issr])
        
        day_quarter_hours = pd.date_range(start = datetime.datetime.strptime('01-03-2015 00:00', '%d-%m-%Y %H:%M'), 
                                                  end = datetime.datetime.strptime('01-03-2015 23:45', '%d-%m-%Y %H:%M'),
                                                  freq = '15min').to_pydatetime().tolist()
        
        cirrus_cover = np.reshape(self.cirrus_cover, (31, -1, self.dim_lon, self.dim_lat))
        cirrus_cover = np.mean(cirrus_cover, axis = (2, 3))
        
        mean_cirrus_cover = np.nanmean(cirrus_cover, axis = 0)
        
        conf_int = lambda x: st.t.interval(alpha=0.95, df=len(x)-1, loc=np.nanmean(x), scale=st.sem(x, nan_policy='omit'))
        
        ci_cirrus_lower, ci_cirrus_upper = zip(*[conf_int(row) for row in cirrus_cover.T])
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(hours, issr_mean, color = 'g', label = 'ISSR (%)')
        ax1.fill_between(hours, ci_lower, 
                          ci_upper, color='g', alpha=.1)
        ax2.plot(day_quarter_hours, mean_cirrus_cover, color = 'r', label = 'daily mean cirrus cover')
        ax2.fill_between(day_quarter_hours, ci_cirrus_lower, 
                          ci_cirrus_upper, color='r', alpha=.1)
        myFmt = mdates.DateFormatter('%H:%M')
        ax1.xaxis.set_major_formatter(myFmt)
        
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)
        ax2.grid(False)
        ax1.set_xlabel('time')
        ax1.set_ylabel('ISSR (%)')
        ax2.set_ylabel('cirrus cover')