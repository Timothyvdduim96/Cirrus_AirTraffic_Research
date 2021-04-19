# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 15:22:10 2021

@author: T Y van der Duim
"""

'''
------------------------------PACKAGES-----------------------------------------
'''

import numpy as np
from SPACE_processing import *
from AIRCRAFT import *
from miscellaneous import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from netCDF4 import Dataset
from datetime import datetime
from scipy import stats
import math as m
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import sys
import matplotlib.dates as mdates
import scipy.stats as st
import jenkspy

'''
---------------------------PLOTTING PREFERENCES--------------------------------
'''

plt.style.use('seaborn-darkgrid')
plt.rc('text', usetex=False)
plt.rc('font', family='times')
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=20) 
plt.rc('font', size=20) 
plt.rc('figure', figsize = (12, 5))

'''
---------------------------------PARAMS----------------------------------------
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
---------------------------------PATHS----------------------------------------
'''

flight_path = 'E:\\Research_cirrus\\Flight_data\\'
lidar_path = 'E:\\Research_cirrus\\CALIPSO_data\\'
meteosat_path = 'E:\\Research_cirrus\\Meteosat_CLAAS_data\\'
meteo_path = 'E:\\Research_cirrus\\ERA5_data\\'

'''
---------------------------------MAIN------------------------------------------
'''

class METEOSAT_analysis:
    
    def __init__(self):
        
        self.quarter_hours = pd.date_range(start = datetime.strptime('01-03-2015 00:00', '%d-%m-%Y %H:%M'), 
                                  end = datetime.strptime('31-03-2015 23:45', '%d-%m-%Y %H:%M'),
                                  freq = '15min').to_pydatetime().tolist()
    
    def validation(self):
        
        lidar = np.load(lidar_path + 'LIDAR_processed\\LIDAR_' + months_complete[0] + '_5km' + '.npy', 
                                        allow_pickle = True)
        
        self.calipso = lidar.item()['calipso']
        self.cirrus_od = lidar.item()['opt_depth']
        self.dates = lidar.item()['dates']
        self.times = lidar.item()['times']
        
        self.overpass_time = [miscellaneous.quarterhr_rounder(datetime.combine(date, time)) for date, time in 
                    zip(self.dates, self.times)]
        
        self.daypasses = pd.DataFrame(np.arange(0, len(self.overpass_time)), index = self.overpass_time,
                     columns = ['index']).between_time('10:00', '15:00')
        
        index = self.daypasses['index'].tolist()
        
        self.overpass_time = [self.overpass_time[idx] for idx in index]
        self.calipso = [self.calipso[idx] for idx in index]
        self.cirrus_od = [self.cirrus_od[idx] for idx in index]
        
        self.flight_over_moment = np.where([any(x == time for x in self.overpass_time) for 
                                            time in self.quarter_hours])[0]
        
        init_meteosat = meteosat()
        init_meteosat.cover_retrieval(cover = True, opt_thick = False, 
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
    
    def validation_plots(self):
        
        CAL_yes_METEO_no = np.where((self.meteosat_ct == 0) & (self.calipso > 0))[0]
        CAL_yes_METEO_yes = np.where((self.meteosat_ct > 0) & (self.calipso > 0))[0]
        CAL_yes_METEO_no_od = self.cirrus_od[CAL_yes_METEO_no]
        CAL_yes_METEO_yes_od = self.cirrus_od[CAL_yes_METEO_yes]
        
        my_dict = dict({'CAL:1, METEOSAT:0': CAL_yes_METEO_no_od, 'CAL:1, METEOSAT:1': CAL_yes_METEO_yes_od})
        od_df = pd.DataFrame.from_dict(my_dict, orient = 'index')
        od_df = od_df.transpose()
        
        plt.figure()
        sns.boxplot(data = od_df)
        
        plt.figure()
        od_df['CAL:1, METEOSAT:0'].hist(density = 1, histtype = 'stepfilled', alpha = .5, 
                                        bins = 100, label = 'CAL:1, METEOSAT:0')
        od_df['CAL:1, METEOSAT:1'].hist(density = 1, histtype = 'stepfilled', alpha = .5, 
                                        color = sns.desaturate("indianred", .75),
                                        bins = 100, label = 'CAL:1, METEOSAT:1')
        plt.legend()
        plt.xlabel('optical depth cirrus')
        plt.ylabel('normalized probability density')
        
    def cirrus_dynamics(self):
        # generate datetimes in steps of 15mins during the day only (between 7AM and 6PM)
        self.quarter_hours_day = pd.DataFrame(np.arange(0, len(self.quarter_hours)), 
                                              index = self.quarter_hours, 
                                              columns = ['index']).between_time('07:00', '18:00')
        
        index = self.quarter_hours_day['index'].tolist()
        
        self.quarter_hours_day = [self.quarter_hours[idx] for idx in index]
        
        # retrieve meteosat data
        init_meteosat = meteosat()
        init_meteosat.cover_retrieval(cover = True, opt_thick = False, 
                                                    idx_list = index)
        
        self.cirrus_cover = init_meteosat.ct_sample
        #self.cirrus_opt_depth = init_meteosat.cot_sample
        self.dates_meteo = init_meteosat.dates_meteo
        
        if len(self.quarter_hours_day) != len(self.dates_meteo):
            missing_times = set(self.quarter_hours_day).difference(self.dates_meteo)
            indices_missing = [self.quarter_hours_day.index(x) for x in missing_times]
            nanarray = np.full([int((max_lon - min_lon) / res_lon), int((max_lat - min_lat) / res_lat)], np.nan)
            
            self.cirrus_cover = np.insert(self.cirrus_cover, indices_missing, nanarray, axis = 0)
            #self.cirrus_opt_depth = np.insert(self.cirrus_opt_depth, indices_missing, nanarray, axis = 0)
        
        # get difference of cirrus cover between two timestamps each day
        nr_days = 31
        
        dayslices = np.arange(0, len(self.cirrus_cover), int(len(self.cirrus_cover) / nr_days))
        dayslices = np.concatenate([dayslices, [None]])
        
        self.diffs = [np.diff(self.cirrus_cover[dayslices[idx]:dayslices[idx +  1], :, :], axis = 0) for 
                      idx in range(len(dayslices) - 1)]
        
        self.diffs = np.concatenate(self.diffs, axis = 0)
        
        #self.diffs_od = [np.diff(self.cirrus_opt_depth[dayslices[idx]:dayslices[idx +  1], :, :], axis = 0) for 
        #              idx in range(len(dayslices) - 1)]
        
        #self.diffs_od = np.concatenate(self.diffs_od, axis = 0)
                
    def flights_Mar15(self, mode):
        
        if mode == 'load':
            self.flights_ATD = np.load(flight_path + 'flights_ATD_03_15.npy', allow_pickle = True)
        elif mode == 'save':
            flights = flight_analysis('03_15')
            flights.exe_ac('load', 'None', 'None',
                       'None','None', 'None','None',
                       'None', 'None', '_full')
            flights.grid_ATD(quarter_hours_day, 15)
            self.flights_ATD = flights.flights_ATD
            
    def delta_cirr_ATD(self):
        
        # reshape ATD array into daily chunks $ omit last 2D array of each day (no Delta cirrus to compare with)
        flights_ATD_res = np.reshape(self.flights_ATD, (-1,31,200,100))
        flights_ATD_res = flights_ATD_res[:-1, :, :, :]
        self.flights_ATD_res = np.reshape(flights_ATD_res, (-1, 200, 100))
        
    def meteo_arrays(self, method):
        
        ERA_Data = miscellaneous.netcdf_import(meteo_path + 'ERA5_' + '03_15' + '.nc')
        ERA_temp = ERA_Data['t']
        ERA_relhum = ERA_Data['r']
        ERA_relhum = np.transpose(ERA_relhum, (0, 1, 3, 2))[:, :, :-1, :-1] 
        ERA_temp = np.transpose(ERA_temp, (0, 1, 3, 2))[:, :, :-1, :-1]
        
        if method == 'mean':
            ERA_temp = np.mean(ERA_temp, axis = 1)
            ERA_relhum = np.mean(ERA_relhum, axis = 1)
        
        elif method == 'max_RH':
            ERA_relhum_vertmax = []
            ERA_temp_vertmax = []
            
            ERA_relhum_max = np.argmax(ERA_relhum, axis = 1)
            ERA_relhum = np.reshape(ERA_relhum, (-1, 200, 100))
            ERA_temp = np.reshape(ERA_temp, (-1, 200, 100))
            
            for i in range(0, len(ERA_relhum), 7):
                full_data_RH = ERA_relhum[i:int(i + 7)]
                full_data_temp = ERA_temp[i:int(i + 7)]
                ERA_relhum_vertmax.append(miscellaneous.multidim_slice(ERA_relhum_max[int(i / 7)], full_data_RH))
                ERA_temp_vertmax.append(miscellaneous.multidim_slice(ERA_relhum_max[int(i / 7)], full_data_temp))
                
            ERA_relhum = np.stack(ERA_relhum_vertmax)
            ERA_temp = np.stack(ERA_temp_vertmax)
        
        # meteo per 15mins instead of hours (repeat 4x)
        ERA_temp = np.repeat(ERA_temp, 4, axis = 0)
        ERA_relhum = np.repeat(ERA_relhum, 4, axis = 0)
        
        day_quarter_hours = pd.date_range(start = datetime.strptime('01-03-2015 07:00', '%d-%m-%Y %H:%M'), 
                                                  end = datetime.strptime('01-03-2015 18:00', '%d-%m-%Y %H:%M'),
                                                  freq = '15min').to_pydatetime().tolist()
        
        self.idx = [key for key, val in enumerate(self.quarter_hours) if val in self.quarter_hours_day]
        
        ERA_temp = ERA_temp[self.idx]
        ERA_relhum = ERA_relhum[self.idx]
        
        ERA_temp = np.reshape(ERA_temp, (-1, 31, 200, 100))
        ERA_temp = ERA_temp[:-1, :, :, :]
        self.ERA_temp = np.reshape(ERA_temp, (-1, 200, 100))
        ERA_relhum = np.reshape(ERA_relhum, (-1, 31, 200, 100))
        ERA_relhum = ERA_relhum[:-1, :, :, :]
        self.ERA_relhum = np.reshape(ERA_relhum, (-1, 200, 100))
            
    def exe_meteo(self):
        self.cirrus_dynamics()
        self.flights_Mar15(mode = 'load')
        self.delta_cirr_ATD()
        self.meteo_arrays(method = 'max_RH')
        
    '''
    -----------------------------VISUALS--------------------------------------
    '''
        
    def ERA5_plot(self):
        
        hours = pd.date_range(start = datetime.strptime('01-03-2015 00:00', '%d-%m-%Y %H:%M'), 
                                  end = datetime.strptime('31-03-2015 23:00', '%d-%m-%Y %H:%M'),
                                  freq = 'H').to_pydatetime().tolist()

        ERA_Data = miscellaneous.netcdf_import(meteo_path + 'ERA5_' + '03_15' + '.nc')
        ERA_temp = ERA_Data['t']
        ERA_relhum = ERA_Data['r']
        ERA_temp = np.mean(ERA_temp, axis = (1, 2, 3))
        ERA_relhum = np.mean(ERA_relhum, axis = (1, 2, 3))
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(hours, ERA_temp, color = 'r', label = 'Temperature')
        ax2.plot(hours, ERA_relhum, color = 'b', label = 'Relative Humidity')
        
        locator = mdates.AutoDateLocator(minticks=31, maxticks=31)
        formatter = mdates.ConciseDateFormatter(locator)
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)
        
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=9, frameon = True)
        ax2.grid(False)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Temperature (K)')
        ax2.set_ylabel('RH (%)')

    def ATD_vs_CIRR(self):
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(self.quarter_hours_day, np.mean(self.flights_ATD, axis = (1, 2)), color = 'r', 
                 label = 'mean Air Traffic Density')
        ax2.plot(self.quarter_hours_day, np.mean(self.cirrus_cover, (1,2)), color = 'b', 
                 label = 'mean cirrus cover')
        
        locator = mdates.AutoDateLocator(minticks=31, maxticks=31)
        formatter = mdates.ConciseDateFormatter(locator)
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)
            
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)
        ax2.grid(False)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('ATD')
        ax2.set_ylabel('Cover')
    
    def plot_cirrus(self):
        
        plt.figure()
        plt.plot(self.quarter_hours_day, np.mean(self.cirrus_cover, (1,2)))

    def daily_cycle(self):
        
        day_quarter_hours = pd.date_range(start = datetime.strptime('01-03-2015 07:00', '%d-%m-%Y %H:%M'), 
                                          end = datetime.strptime('01-03-2015 18:00', '%d-%m-%Y %H:%M'),
                                          freq = '15min').to_pydatetime().tolist()
        
        #day_quarter_hours = [datestamp.time() for datestamp in day_quarter_hours]
        mean_flights_ATD = np.mean(self.flights_ATD, axis = 0)
        ci_flights = 1.96 * np.std(self.flights_ATD, axis = 0)
        mean_cirrus_cover = np.mean(self.cirrus_cover, axis = 0)
        ci_cirrus = 1.96 * np.std(self.cirrus_cover, axis = 0)
        
        conf_int = lambda x: st.t.interval(alpha=0.95, df=len(x)-1, loc=np.mean(x), scale=st.sem(x))
        
        ci_ATD_lower, ci_ATD_upper = zip(*[conf_int(row) for row in flights_ATD.T])
        ci_cirrus_lower, ci_cirrus_upper = zip(*[conf_int(row) for row in self.cirrus_cover.T])
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(day_quarter_hours, mean_flights_ATD, color = 'b', label = 'Air Traffic Density')
        ax1.fill_between(day_quarter_hours, ci_ATD_lower, 
                         ci_ATD_upper, color='b', alpha=.1)
        ax2.plot(day_quarter_hours, mean_cirrus_cover, color = 'r', label = 'Cirrus cover')
        ax2.fill_between(day_quarter_hours, ci_cirrus_lower, 
                         ci_cirrus_upper, color='r', alpha=.1)
        myFmt = mdates.DateFormatter('%H:%M')
        ax1.xaxis.set_major_formatter(myFmt)
        
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)
        ax2.grid(False)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Daily mean ATD')
        ax2.set_ylabel('Daily mean cirrus cover')
        
    def animation_ATD(self, save):
        
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
        
        lons, lats = map(*np.meshgrid(np.arange(-9.875, 40.125, 0.25), np.arange(35.125, 60.125, 0.25)))
        
        # contourf 
        im = map.contourf(lons, lats, flights_ATD[0].T, np.arange(0, 20.1, 0.1), extend='max', cmap='binary')
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
            im = map.contourf(lons, lats, flights_ATD[i].T, np.arange(0, 20.1, 0.1), extend='max', cmap='binary')
            title = ax.text(0.5,1.05,"ATD\n{0}".format(str(quarter_hours_day[i])),
                            ha="center", transform=ax.transAxes,)
        
        myAnimation = animation.FuncAnimation(fig, animate, frames = len(flights_ATD))
        
        if save == True:
            # Set up formatting for the movie files
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
            myAnimation.save('ATD.mp4', writer=writer)
    
#%%
df = pd.DataFrame(list(zip(np.reshape(meteo.flights_ATD_res, -1), 
                            np.reshape(meteo.diffs, -1), np.reshape(meteo.ERA_temp, -1),
                            np.reshape(meteo.ERA_relhum, -1))), columns = ['ATD', 'delta_cirrus', 'temp', 'RH'])

#%%

df_adj = df[df['RH'] >= 100]# & (df['RH'] < 52) & (df['temp'] > 222) & (df['temp'] < 223)]
break_1 = []
break_2 = []
break_3 = []
break_4 = []

for hr in range(45):
    print(hr)
    breaks = jenkspy.jenks_breaks(np.reshape(meteo.flights_ATD_res[hr, :, :], -1), nb_class = 5)
    break_1.append(breaks[1])
    break_2.append(breaks[2])
    break_3.append(breaks[3])
    break_4.append(breaks[4])
    
day_quarter_hours = pd.date_range(start = datetime.strptime('01-03-2015 07:00', '%d-%m-%Y %H:%M'), 
                                          end = datetime.strptime('01-03-2015 18:00', '%d-%m-%Y %H:%M'),
                                          freq = '15min').to_pydatetime().tolist()

day_quarter_hours = [t.strftime('%H:%M') for t in day_quarter_hours]

break_points = pd.DataFrame(list(zip(break_1, break_2, break_3, break_4, [50] * 45)), 
                            index = day_quarter_hours)
break_points.loc['mean'] = break_points.mean()
break_points = break_points.reindex(index=break_points.index[::-1])

fig, ax = plt.subplots()
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=12)
break_points.plot(kind = 'barh', stacked=True, ax = ax)
plt.xlabel('Air Traffic Density (km km$^{-2}$ hr$^{-1}$)')
ax.yaxis.set_tick_params(rotation=30)
legend = plt.legend(frameon = 1)
frame = legend.get_frame()
frame.set_color('white')
plt.xlim([0, 40])

breaks = break_points.loc['mean'].tolist()
breaks.insert(0, 0)
breaks[-1] = 1e3

df_adj['cut_jenks'] = pd.cut(df_adj['ATD'],
                        bins = breaks,
                        labels = ['no-very low ATD', 'low ATD', 'moderate ATD', 'high ATD', 'very high ATD'],
                        include_lowest = True)

#['no-low ATD', 'moderate ATD', 'high ATD']
print(df_adj['cut_jenks'].value_counts())

df_adj.groupby(by = 'cut_jenks').mean()

