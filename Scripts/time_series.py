# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 13:07:04 2021

@author: T Y van der Duim
"""

'''
------------------------------PACKAGES-----------------------------------------
'''

import numpy as np
from SPACE_processing import *
from AIRCRAFT import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from netCDF4 import Dataset
from datetime import datetime
from scipy import stats
import math as m
from sklearn.linear_model import LinearRegression

def netcdf_import(file):
    data = Dataset(file,'r')
    #print(data.variables)
    #print(data.variables.keys())
    my_dict = {}
    for key in data.variables.keys():
        if key == 'longitude' or key == 'latitude' or key == 'time' or key == 'level':
            my_dict[key] = data[key][:]
        else:
            my_dict[key] = data[key][:, :, :, :]
    return my_dict

'''
---------------------------------PARAMS----------------------------------------
'''

res_lon = 0.25 # set desired resolution in longitudinal direction in degs
res_lat = 0.25 # set desired resolution in latitudinal direction in degs
min_lon = -10 # research area min longitude
max_lon = 40 # research area max longitude
min_lat = 35 # research area min latitude
max_lat = 60 # research area max latitude

#%%

def time_map(dates, times):

        # get all overpass times over Europe of CALIPSO
        overpass_time = [datetime.combine(date, time) for date, time in zip(dates, times)]
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
        print(overpass_freq)
        ax.bar((overpass_freq.index / 24 + 1 / 48 ) * 2 * m.pi, 
               overpass_freq["nr of CALIPSO overpasses during March '15"], 
               width = 0.2, alpha = 0.5, color = "#f39c12")
        
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=14)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks(np.arange(0, 2 * m.pi + 2 / 24 * m.pi, 2 / 24 * m.pi))
        ax.set_yticks(np.arange(0, 600, 100))
        ax.set_title("Overpass frequency of CALIPSO over Europe during Mar, Jun,\n Sep & Dec from 2015 till 2020")
        ticks = [f"{i}:00" for i in range(0, 24, 1)]
        ax.set_xticklabels(ticks)
        
#%%
months = ['03_15', '06_15', '09_15', '12_15', '03_16', '06_16', '09_16', '12_16',
          '03_17', '06_17', '09_17', '12_17', '03_18', '06_18', '09_18', '12_18',
          '03_19', '06_19', '09_19', '12_19', '03_20', '06_20', '09_20', '12_20']

calipso_time = []
calipso_nighttime = []
calipso_daytime = []

atd_time = []
alldates = []
alltimes = []

rel_hum_series = []
temp_series = []

mode = 'load'


for month in months:
    print(month)
    
    if mode == 'save':
        lidar = CALIPSO_analysis(lidar_path + 'LIDAR_' + month, 15, False)
        
        savearray = {'calipso': lidar.CALIPSO_cirrus, 'dates': lidar.dates, 'times': lidar.times,
                      'lon_pos': lidar.lon_pos, 'lat_pos': lidar.lat_pos}
        
        calipso = lidar.CALIPSO_cirrus
        
        #np.save('LIDAR_' + month, savearray, allow_pickle = True)
        
        # flatten arrays and remove nan instances
        calipso_1d = np.reshape(calipso, -1)
        calipso_time.append(calipso_1d[~(np.isnan(calipso_1d))])
                
        if eval(month[-2:]) < 19:
    
            flight = flight_analysis('Flights_20{0}{1}_20{0}{1}.csv'.format(month[-2:], month[:2]), 30,
                                     'load', 'Flights_20{0}{1}_20{0}{1}.pkl'.format(month[-2:], month[:2]),
                                         dates = calipso.dates, times = calipso.times, 
                                         lon_pos = calipso.lon_pos, 
                                         lat_pos = calipso.lat_pos)
    
            atd = flight.flights_ATD
            calipso_1d = np.reshape(calipso.calipso, -1)
            atd_1d = np.reshape(atd, -1)
            atd_1d = np.where(np.isnan(calipso_1d) == True, np.nan, atd_1d)
            atd_1d = atd_1d[~(np.isnan(atd_1d))]
            np.save('Flights_' + month, atd_1d, allow_pickle = True)
            atd_time.append(atd_1d)
            
    elif mode == 'load':
            calipso = np.load(lidar_path + 'LIDAR_processed\\LIDAR_' + month + '.npy', allow_pickle = True)
            alldates.append(calipso.item()['dates'])
            alltimes.append(calipso.item()['times'])
            
            overpass_time = [datetime.combine(date, time) for date, time in 
                             zip(calipso.item()['dates'], calipso.item()['times'])]
            
            nightpasses = pd.DataFrame(np.arange(0, len(overpass_time)), index = overpass_time,
                         columns = ['index']).between_time('22:00', '5:00')
            
            mask = np.zeros(len(calipso.item()['calipso']),dtype=bool)
            mask[nightpasses] = True
            calipso_night = calipso.item()['calipso'][mask]
            calipso_day = calipso.item()['calipso'][~mask]
            
            print(np.shape(calipso.item()['calipso']), np.shape(calipso_night), np.shape(calipso_day))
            
            calipso_total = np.reshape(calipso.item()['calipso'], -1)
            calipso_night = np.reshape(calipso_night, -1)
            calipso_day = np.reshape(calipso_day, -1)
            calipso_time.append(calipso_total[~(np.isnan(calipso_total))])
            calipso_nighttime.append(calipso_night[~(np.isnan(calipso_night))])
            calipso_daytime.append(calipso_day[~(np.isnan(calipso_day))])
            
            if eval(month[-2:]) < 19:
                atd = np.load(path + 'Flight_data\\Flights_{0}'.format(month) + '.npy',
                          allow_pickle = True)
                atd_time.append(atd)
            
            ERA_Data = netcdf_import('E:\\Research_cirrus\\ERA5_data\\' + 'ERA5_' + month + '.nc')
            ERA_temp = ERA_Data['t']
            ERA_relhum = ERA_Data['r']
            
            # select parts of ERA5 data at times when CALIPSO overpasses Europe
            overpass_time = [datetime.combine(date, time) 
                              for date, time in zip(calipso.item()['dates'], calipso.item()['times'])] # merge dates and times to get a datetime list of Calipso overpasses
            
            def hour_rounder(t):
                # Rounds to nearest hour by adding a timedelta hour if minute >= 30
                return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
                           +timedelta(hours=t.minute//30))
            
            start = overpass_time[0]
            end = overpass_time[-1]
            
            hour_list = pd.date_range(start = hour_rounder(start), end = hour_rounder(end),
                                      freq = 'H').to_pydatetime().tolist()
            
            overpass_time = [hour_rounder(overpass) for overpass in overpass_time]
            
            idx = [key for key, val in enumerate(hour_list) if val in overpass_time]
            
            if idx[-1] >= len(ERA_relhum):
                idx[-1] = idx[-1] - 1
                
            rel_hum = ERA_relhum[idx, :, :, :]
            temp = ERA_temp[idx, :, :, :]
            
            def duplicates(lst, item):
                return [i for i, x in enumerate(lst) if x == item]
        
            remove_dup = dict((x, duplicates(overpass_time, x))[1] for x in set(overpass_time) 
                              if overpass_time.count(x) > 1)
            
            cirrus_cover = np.isnan(calipso.item()['calipso'])
            if len(idx) != len(cirrus_cover):
                cirrus_cover = np.delete(cirrus_cover, list(remove_dup.values()), axis = 0)
            else:
                cirrus_cover = cirrus_cover
            
            rel_hum = np.transpose(rel_hum, (0, 1, 3, 2))[:, :, :-1, :-1] 
            temp = np.transpose(temp, (0, 1, 3, 2))[:, :, :-1, :-1]
            rel_hum = np.mean(rel_hum, axis = 1)
            temp = np.mean(temp, axis = 1)
            rel_hum = np.where(cirrus_cover == True, np.nan, rel_hum)
            temp = np.where(cirrus_cover == True, np.nan, temp)     
            
            rel_hum = rel_hum.reshape(-1)
            temp = temp.reshape(-1)
            
            rel_hum_series.append(rel_hum[~(np.isnan(rel_hum))])
            temp_series.append(temp[~(np.isnan(temp))])

#time_map(np.concatenate(alldates), np.concatenate(alltimes))

#%%

rh_bin_min = 20
rh_bin_max = 30
t_bin_min = 220
t_bin_max = 230

rh_binned = []
t_binned = []
calipso_binned = []
temp_rh_grid = []

for month in range(len(months)):
    stat = stats.binned_statistic_2d(rel_hum_series[month], temp_series[month],
                                     [1] * len(temp_series[month]), statistic = 'sum',
                                     bins = [int((130 - 0) / 10), int((250 - 190) / 10)],
                                     range = [[0, 130], [190, 250]])
    
    temp_rh_grid.append(stat.statistic)
    
    print(temp_series[month])
    selection = np.where((rel_hum_series[month] > rh_bin_min) & (rel_hum_series[month] < rh_bin_max) &
                         (temp_series[month] > t_bin_min) & (temp_series[month] < t_bin_max))
    
    if len(selection[0]) == 0:
        print("bin too small")
        break
    
    rh_binned.append(rel_hum_series[month][selection])
    t_binned.append(temp_series[month][selection])
    calipso_binned.append(calipso_time[month][selection])

checkstack = np.stack(temp_rh_grid)
temp_rh_grid = np.min(np.stack(temp_rh_grid), axis = 0)
statpd = pd.DataFrame(temp_rh_grid, index = list(map(str, stat.x_edge))[:-1],
                          columns = list(map(str, stat.y_edge))[:-1])

#%%
'''
------------------------TIME SERIES CALIPSO------------------------------------
'''

mean = lambda x: np.nanmean(x)

mean_cover = list(map(mean, calipso_binned))

dates = [month.replace('_', '-') for month in months]
years = ['2015', '2016', '2017', '2018', '2019', '2020']

def pivot_array(arr):
    cover_df = pd.DataFrame(list(zip(dates, arr)), columns = ['dates', 'mean_cirrus_cover'])
    cover_df['dates'] = pd.to_datetime(cover_df['dates'], format='%m-%y')
    cover_df['month'] = cover_df['dates'].apply(lambda x: x.strftime("%m"))
    cover_df['year'] = cover_df['dates'].apply(lambda x: x.strftime("%Y"))
    cover_df = cover_df.drop(['dates'], axis = 1)
    pivot_cirrus = pd.pivot_table(cover_df, values="mean_cirrus_cover",
                                    index=["month"],
                                    columns=["year"],
                                    fill_value=0,
                                    margins=True)

    pivot_cirrus.index = ['Mar', 'Jun', 'Sep', 'Dec', 'All']
    return pivot_cirrus

cirrus = pivot_array(mean_cover)

#cover_df = cover_df.set_index('month', append=True)

plt.figure()
ax = sns.heatmap(cirrus, cmap='seismic', robust=True, fmt='.2f', 
                  annot=True, linewidths=.5, annot_kws={'size':11}, 
                  cbar_kws={'shrink':.8, 'label':'Mean cirrus coverage'})                       
    
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)

#%%
calipso_nighttime = list(map(mean, calipso_nighttime))
calipso_daytime = list(map(mean, calipso_daytime))

calipso_difference = [(night - day) / day * 100 for day, night in zip(calipso_daytime, calipso_nighttime)]
calipso_difference = pivot_array(calipso_difference)

plt.figure()
ax = sns.heatmap(calipso_difference, cmap='seismic', center = 0, robust=True, fmt='.2f', 
                  annot=True, linewidths=.5, annot_kws={'size':11}, 
                  cbar_kws={'shrink':.8, 'label':'$\Delta$ cirrus (%)'})                       
plt.title('night compared to day')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)

#%%
'''
--------------------------TIME SERIES ATD--------------------------------------
'''

atdmeans = [np.mean(atd) for atd in atd_time]
years = ['2015', '2016', '2017', '2018']

atd_df = pd.DataFrame(list(zip(dates, atdmeans)), columns = ['dates', 'mean_atd'])
atd_df['dates'] = pd.to_datetime(atd_df['dates'], format='%m-%y')
atd_df['month'] = atd_df['dates'].apply(lambda x: x.strftime("%m"))
atd_df['year'] = atd_df['dates'].apply(lambda x: x.strftime("%Y"))
atd_df = atd_df.drop(['dates'], axis = 1)

cirrus = pd.pivot_table(atd_df, values="mean_atd",
                                    index=["month"],
                                    columns=["year"],
                                    fill_value=0,
                                    margins=True)

cirrus.index = ['Mar', 'Jun', 'Sep', 'Dec', 'All']

plt.figure()
ax = sns.heatmap(cirrus, cmap='plasma', robust=True, fmt='.2f', 
                  annot=True, linewidths=.5, annot_kws={'size':11}, 
                  cbar_kws={'shrink':.8, 'label':'Mean Air Traffic Density'})                       
    
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)

#%%
'''
------------------------SCATTER FLIGHTS VS ATD---------------------------------
'''

nr_flights = []

for month in months:
    print(month)
    if eval(month[-2:]) < 19:
            calipso = np.load(lidar_path + 'LIDAR_processed\\LIDAR_' + month + '.npy', allow_pickle = True)
            flight = flight_analysis('Flights_20{0}{1}_20{0}{1}.csv'.format(month[-2:], month[:2]), 30,
                                     'load', 'Flights_20{0}{1}_20{0}{1}.pkl'.format(month[-2:], month[:2]),
                                         dates = calipso.item()['dates'], times = calipso.item()['times'], 
                                         lon_pos = calipso.item()['lon_pos'], 
                                         lat_pos = calipso.item()['lat_pos'])
            nr_flights.append(len(flight.flights))

corr = round(np.corrcoef(nr_flights, atdmeans)[0,1]**2, 2)

plt.figure()
ax = sns.regplot(atdmeans, nr_flights)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.xlabel('Air traffic density (km/(km$\cdot$hr))')
plt.ylabel('Nr of monthly flights')
ax.text(1, 1e6, '$r^2$ = ' + str(corr), fontsize = 20)
for i, txt in enumerate(dates[:16]):
    ax.annotate(txt, (atdmeans[i], nr_flights[i]), fontsize = 12)
plt.show()

#%%
'''
------------------------LINEAR REGRESSION MODEL--------------------------------
'''
X = np.array(nr_flights).reshape(-1, 1)

reg = LinearRegression().fit(X, atdmeans)

reg.score(X, atdmeans)

flights_1920 = []

for date in months[16:]:
    print(date)
    month = date[0:2]
    year = date[-2:]
    flights_1920.append(len(flights_after_2018(month, year).flights0319_combined))

predicted_atd = reg.predict(np.array(flights_1920).reshape(-1, 1))
predicted_atd = np.where(predicted_atd < 0, 0, predicted_atd)

atdmeans = np.concatenate([atdmeans, predicted_atd])

years = ['2015', '2016', '2017', '2018', '2019', '2020']

atd_df = pd.DataFrame(list(zip(dates, atdmeans)), columns = ['dates', 'mean_atd'])
atd_df['dates'] = pd.to_datetime(atd_df['dates'], format='%m-%y')
atd_df['month'] = atd_df['dates'].apply(lambda x: x.strftime("%m"))
atd_df['year'] = atd_df['dates'].apply(lambda x: x.strftime("%Y"))
atd_df = atd_df.drop(['dates'], axis = 1)

pivot_cirrus = pd.pivot_table(atd_df, values="mean_atd",
                                    index=["month"],
                                    columns=["year"],
                                    fill_value=0,
                                    margins=True)

pivot_cirrus.index = ['Mar', 'Jun', 'Sep', 'Dec', 'All']

plt.figure()
ax = sns.heatmap(pivot_cirrus, cmap='seismic', robust=True, fmt='.2f', 
                  annot=True, linewidths=.5, annot_kws={'size':11}, 
                  cbar_kws={'shrink':.8, 'label':'Mean Air Traffic Density'})                       
    
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)

#%%
'''
-----------------------------TIME SERIES RH------------------------------------
'''
rel_hum_series = list(map(mean, rel_hum_series))

relhum = pivot_array(rel_hum_series)

#cover_df = cover_df.set_index('month', append=True)

plt.figure()
ax = sns.heatmap(relhum, cmap='viridis', robust=True, fmt='.2f', 
                  annot=True, linewidths=.5, annot_kws={'size':11}, 
                  cbar_kws={'shrink':.8, 'label':'Mean upper tropospheric RH'})                       
    
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)

#%%
'''
----------------------------TIME SERIES TEMP-----------------------------------
'''
temp_series = list(map(mean, temp_series))

temp = pivot_array(temp_series)

#cover_df = cover_df.set_index('month', append=True)

plt.figure()
ax = sns.heatmap(temp, cmap='viridis', robust=True, fmt='.2f', 
                  annot=True, linewidths=.5, annot_kws={'size':11}, 
                  cbar_kws={'shrink':.8, 'label':'Mean upper tropospheric temperature ($^{\circ}$C)'})                       
    
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)

corrs = pd.DataFrame(list(zip(temp_series, rel_hum_series, mean_cover)), 
                     columns = ['T', 'RH', 'cirrus cover']).corr()
