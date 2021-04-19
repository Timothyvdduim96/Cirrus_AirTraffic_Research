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
import random
import matplotlib

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
meteo_path = 'E:\\Research_cirrus\\ERA5_data\\'

#%%
'''
----------------------------------MAIN-----------------------------------------
'''
all_calipso = []

for month in months_complete:
    print(month)
    lidar = CALIPSO_analysis(lidar_path + 'LIDAR_' + month, 15, True)
    calipso = lidar.layered_cirrus
    calipso = np.vstack(calipso)
    all_calipso.append(calipso[np.isnan(calipso) == False])

#%%
fig = plt.figure()
ax1 = fig.add_subplot(111)

for calipso in all_calipso:
    sns.kdeplot(calipso, vertical = True)
    
plt.ylim([np.min(all_calipso), np.max(all_calipso)])
ax1.set_yscale('log')
ax1.set_ylim(10 * np.ceil(calipso.max() / 10), calipso.min()) # avoid truncation of 1000 hPa
subs = [1,2,5]

if calipso.max() / calipso.min() < 30:
    subs = [1,2,3,4,5,6,7,8,9]
    
y1loc = matplotlib.ticker.LogLocator(base=10, subs=subs)
ax1.yaxis.set_major_locator(y1loc)
fmt = matplotlib.ticker.FormatStrFormatter("%g")
ax1.yaxis.set_major_formatter(fmt)
#plt.gca().invert_yaxis()
z0 = 8.400    # scale height for pressure_to_altitude conversion [km]
altitude = z0 * np.log(1015.23/y)
# add second y axis for altitude scale 
axr = ax1.twinx()
label_xcoor = 1.05
axr.set_ylabel("Altitude [km]")
axr.yaxis.set_label_coords(label_xcoor, 0.5)
axr.set_ylim(altitude.min(), altitude.max())
yrloc = matplotlib.ticker.MaxNLocator(steps=[1,2,5,10])
axr.yaxis.set_major_locator(yrloc)
axr.yaxis.tick_right()
plt.show()

#%%

class time_series:
                 
    def CALIPSO(self, month, mode):
                
        if mode == 'save':
            
            lidar = CALIPSO_analysis(lidar_path + 'LIDAR_' + month, 15, False)
            self.calipso = lidar.CALIPSO_cirrus
            self.calipso_od = lidar.cirrus_od
            self.calipso_od = np.where(np.isnan(self.calipso) == True, np.nan, self.calipso_od)
            self.calipso_od = miscellaneous.flatten_clean(self.calipso_od)
            
            savearray = {'calipso': self.calipso, 'dates': lidar.dates, 'times': lidar.times,
                          'lon_pos': lidar.lon_pos, 'lat_pos': lidar.lat_pos, 'opt_depth': lidar.cirrus_od}

            np.save('LIDAR_' + month, savearray, allow_pickle = True)          
                
        elif mode == 'load':
                lidar = np.load(lidar_path + 'LIDAR_processed\\LIDAR_' + month + '.npy', 
                                allow_pickle = True)
                self.calipso = lidar.item()['calipso']
                try:
                    self.calipso_od = lidar.item()['opt_depth']
                    self.calipso_od = np.where(np.isnan(self.calipso) == True, np.nan, self.calipso_od)
                    self.calipso_od = miscellaneous.flatten_clean(self.calipso_od)
                except:
                    pass
                
        else:
            sys.exit('mode is not recognized')
        
        try:
            self.dates = lidar.item()['dates']
            self.times = lidar.item()['times']
            self.lon_pos = lidar.item()['lon_pos']
            self.lat_pos = lidar.item()['lat_pos']
        except:
            self.dates = lidar.dates
            self.times = lidar.times
            self.lon_pos = lidar.lon_pos
            self.lat_pos = lidar.lat_pos
            
        self.overpass_time = [datetime.combine(date, time) for date, time in 
                        zip(self.dates, self.times)]
            
        nightpasses = pd.DataFrame(np.arange(0, len(self.overpass_time)), index = self.overpass_time,
                     columns = ['index']).between_time('22:00', '5:00')
        
        mask = np.zeros(len(self.calipso),dtype=bool)
        mask[nightpasses] = True
        calipso_night = self.calipso[mask]
        calipso_day = self.calipso[~mask]            
                
        self.calipso_total = miscellaneous.flatten_clean(self.calipso)
        self.calipso_night = miscellaneous.flatten_clean(calipso_night)
        self.calipso_day = miscellaneous.flatten_clean(calipso_day)
        
    def air_traffic(self, month, mode):
        
        if month in months_complete:
            
            if mode == 'save':
                
                flight = flight_analysis(month)
                flight.exe_ac(mode = mode, resample_interval = '1min', window = 30, V_aircraft = 1200,
                                    dates = self.dates, times = self.times, lon_pos = self.lon_pos,
                                    lat_pos = self.lat_pos, savename = 'interpol_03_19')
                flight.grid_ATD(flight.combined_datetime, 30)
                atd = flight.flights_ATD
                atd = np.where(np.isnan(self.calipso) == True, np.nan, atd)
                self.atd = miscellaneous.flatten_clean(atd)
                np.save('Flights_{0}'.format(month), self.atd, allow_pickle = True)
                
            elif mode == 'load':
                
                self.atd = np.load(flight_path + 'Flights_{0}'.format(month) + '.npy',
                          allow_pickle = True)
                
            else:
                sys.exit('mode is not recognized')
                
    def meteo_data(self, month):
        
        ERA_Data = miscellaneous.netcdf_import(meteo_path + 'ERA5_' + month + '.nc')
        ERA_temp = ERA_Data['t']
        ERA_relhum = ERA_Data['r']
        
        hour_list = pd.date_range(start = miscellaneous.hour_rounder(self.overpass_time[0]), 
                                  end = miscellaneous.hour_rounder(self.overpass_time[-1]),
                                  freq = 'H').to_pydatetime().tolist()
        
        self.overpass_time = [miscellaneous.hour_rounder(overpass) for overpass
                              in self.overpass_time]
        
        idx = [key for key, val in enumerate(hour_list) if val in self.overpass_time]
        
        idx[-1] = idx[-1] - 1 if (idx[-1] >= len(ERA_relhum)) else idx[-1]

        rel_hum = ERA_relhum[idx, :, :, :]
        temp = ERA_temp[idx, :, :, :]
    
        remove_dup = dict((x, miscellaneous.duplicates(self.overpass_time, x))[1] for x 
                          in set(self.overpass_time) if self.overpass_time.count(x) > 1)
        
        cirrus_cover = np.isnan(self.calipso)
        
        cirrus_cover = np.delete(cirrus_cover, list(remove_dup.values()), 
                                 axis = 0) if (len(idx) != len(cirrus_cover)) else cirrus_cover

        # reshape arrays to match format cirrus
        rel_hum = np.transpose(rel_hum, (0, 1, 3, 2))[:, :, :-1, :-1] 
        temp = np.transpose(temp, (0, 1, 3, 2))[:, :, :-1, :-1]
        
        # take mean in the vertical
        rel_hum = np.mean(rel_hum, axis = 1)
        temp = np.mean(temp, axis = 1)
        
        # match locations to those where CALIPSO overpasses
        rel_hum = np.where(cirrus_cover == True, np.nan, rel_hum)
        temp = np.where(cirrus_cover == True, np.nan, temp)     
        
        self.rel_hum = miscellaneous.flatten_clean(rel_hum)
        self.temp = miscellaneous.flatten_clean(temp)
        
    def exe(self, mode):
        
        self.calipso_day_list = []
        self.calipso_night_list = []
        self.calipso_total_list = []
        self.atd_list = []
        self.rel_hum_list = []
        self.temp_list = []

        for month in months_complete:
            self.CALIPSO(month, mode)
            self.air_traffic(month, mode) if (month in months_1518) else ''
            self.meteo_data(month)
            self.calipso_day_list.append(self.calipso_day)
            self.calipso_night_list.append(self.calipso_night)
            self.calipso_total_list.append(self.calipso_total)
            self.atd_list.append(self.atd) if (month in months_1518) else ''
            self.rel_hum_list.append(self.rel_hum)
            self.temp_list.append(self.temp)
        
    def bin_meteo(self, rh_bin_min, rh_bin_max, t_bin_min, t_bin_max):
    
        selection = np.where((self.rel_hum > rh_bin_min) & (self.rel_hum < rh_bin_max) &
                         (self.temp > t_bin_min) & (self.temp < t_bin_max))
    
        sys.exit("bin too small") if (len(selection[0]) == 0) else ''
    
        self.rel_hum_binned = self.rel_hum[selection]
        self.temp_binned = self.temp[selection]
        self.calipso_total_binned = self.calipso_total[selection]


    '''
    -----------------------------VISUALS--------------------------------------
    '''

    def time_series_CALIPSO(self, calipso_list):
        
        nanmean = lambda x: np.nanmean(x)
        mean_cover = list(map(nanmean, calipso_list))
        cirrus = miscellaneous.pivot_array(mean_cover, dates)
        
        plt.figure()
        ax = sns.heatmap(cirrus, cmap='seismic', robust=True, fmt='.2f', 
                          annot=True, linewidths=.5, annot_kws={'size':11}, 
                          cbar_kws={'shrink':.8, 'label':'Mean cirrus coverage'})                       
            
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)

    def dayvsnight_CALIPSO(self, calipso_daytime, calipso_nighttime):
        
        nanmean = lambda x: np.nanmean(x)
        calipso_nighttime = list(map(nanmean, calipso_nighttime))
        calipso_daytime = list(map(nanmean, calipso_daytime))
        
        calipso_difference = [(night - day) / day * 100 for day, night in zip(calipso_daytime, calipso_nighttime)]
        self.calipso_difference = miscellaneous.pivot_array(calipso_difference, dates)
        
        plt.figure()
        ax = sns.heatmap(self.calipso_difference, cmap='seismic', center = 0, robust=True, fmt='.2f', 
                          annot=True, linewidths=.5, annot_kws={'size':11}, 
                          cbar_kws={'shrink':.8, 'label':'$\Delta$ cirrus (%)'})                       
        plt.title('night compared to day')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)

    def time_series_ATD(self, atd_list):
        
        atdmeans = [np.mean(atd) for atd in atd_list]
        
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
        
    def scatter_flights_vs_atd(self, atd_list):
        
        atdmeans = [np.mean(atd) for atd in atd_list]
        
        self.nr_flights = []

        for month in months_1518:
            print(month)
            calipso = np.load(lidar_path + 'LIDAR_processed\\LIDAR_' + month + '.npy', allow_pickle = True)
            flight = flight_analysis(month)
            self.nr_flights.append(len(flight.individual_flights()))
        
        corr = round(np.corrcoef(self.nr_flights, atdmeans)[0,1]**2, 2)
        
        plt.figure()
        ax = sns.regplot(self.nr_flights, atdmeans)
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.xlabel('Nr of monthly flights')
        plt.ylabel('Air traffic density (km/(km$\cdot$hr))')
        ax.text(9e5, 0.9, '$r^2$ = ' + str(corr), fontsize = 20)
        for i, txt in enumerate(dates[:16]):
            ax.annotate(txt, (self.nr_flights[i], atdmeans[i]), fontsize = 12)
        plt.show()

    def time_series_flights(self, atd_list):
        
        try:
            self.nr_flights
        except:
            self.scatter_flights_vs_atd(atd_list)
        
        list_of_months = ['Mar', 'Jun', 'Sep', 'Dec', 'Mean'] * (2018 - 2015 + 1)
        list_of_years = list(np.repeat(years, 5))
        mean_year_flights = np.reshape(self.nr_flights, (4, 4)).mean(axis = 1)
        
        # get percentage change air traffic
        AT_change = pd.DataFrame(mean_year_flights).pct_change() * 100
        
        nr_flights_ext = self.nr_flights[:]
        for idx, val in zip([4, 9, 14, 19], mean_year_flights):
            nr_flights_ext.insert(idx, val)
        
        plot_flights = pd.DataFrame(list(zip(nr_flights_ext, list_of_months, list_of_years)),
                                    columns = ['Number of flights', 'Month', 'Year'])
        
        plt.figure(figsize=(10,10))
        ax = sns.lineplot(data = plot_flights, x = 'Year', y = 'Number of flights', 
                          hue = 'Month', marker='o')
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        plt.legend(loc = 2)
        
        for i, txt in enumerate(AT_change[0][1:]):
            txt = '+{0}%'.format(str(round(txt, 1)))
            ax.annotate(txt, (plot_flights['Year'][(i + 1) * 5], mean_year_flights[i + 1] + 1e4), 
                        fontsize = 15, weight = 'bold')
        plt.show()
    
    def linregres_1920(self, atd_list):
                
        try:
            self.nr_flights
        except:
            self.scatter_flights_vs_atd(atd_list)
            
        X = np.array(self.nr_flights).reshape(-1, 1)
        
        atdmeans = [np.mean(atd) for atd in atd_list]
        reg = LinearRegression().fit(X, atdmeans)
        
        flights_1920 = [len(flights_after_2018(date[0:2], date[-2:]).flights0319_combined) for date in months_1920]
        print(flights_1920)
        predicted_atd = reg.predict(np.array(flights_1920).reshape(-1, 1))
        predicted_atd = np.where(predicted_atd < 0, 0, predicted_atd)
        print(predicted_atd)
        atdmeans = np.concatenate([atdmeans, predicted_atd])
        print(atdmeans)
        atd_df = pd.DataFrame(list(zip(dates, atdmeans)), columns = ['dates', 'mean_atd'])
        atd_df['dates'] = pd.to_datetime(atd_df['dates'], format='%m-%y')
        atd_df['month'] = atd_df['dates'].apply(lambda x: x.strftime("%m"))
        atd_df['year'] = atd_df['dates'].apply(lambda x: x.strftime("%Y"))
        self.atd_df = atd_df.drop(['dates'], axis = 1)
        
        pivot_cirrus = pd.pivot_table(self.atd_df, values="mean_atd",
                                            index=["month"],
                                            columns=["year"],
                                            fill_value=0,
                                            margins=True)
        
        pivot_cirrus.index = ['Mar', 'Jun', 'Sep', 'Dec', 'All']
        
        plt.figure()
        ax = sns.heatmap(pivot_cirrus, cmap='plasma', robust=True, fmt='.2f', 
                          annot=True, linewidths=.5, annot_kws={'size':11}, 
                          cbar_kws={'shrink':.8, 'label':'Mean Air Traffic Density'})                       
            
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)

    def influence_plot(self, atd_list):
        
        atdmeans = [np.mean(atd) for atd in atd_list]
        
        lm = sm.OLS(atdmeans, X).fit()
        
        fig, ax = plt.subplots(figsize=(12,8))
        fig = sm.graphics.influence_plot(lm, ax= ax, criterion="cooks")
        
    def time_series_RH(self, rel_hum_list):
        
        nanmean = lambda x: np.nanmean(x)
        rel_hum_series = list(map(nanmean, rel_hum_list))
        relhum = miscellaneous.pivot_array(rel_hum_series, dates)

        plt.figure()
        ax = sns.heatmap(relhum, cmap='viridis', robust=True, fmt='.2f', 
                          annot=True, linewidths=.5, annot_kws={'size':11}, 
                          cbar_kws={'shrink':.8, 'label':'Mean upper tropospheric RH'})                       
            
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
        
    def time_series_T(self, temp_list, rel_hum_list):
        
        nanmean = lambda x: np.nanmean(x)
        temp_series = list(map(nanmean, temp_list))
        temp = miscellaneous.pivot_array(temp_series, dates)
                
        plt.figure()
        ax = sns.heatmap(temp, cmap='viridis', robust=True, fmt='.2f', 
                          annot=True, linewidths=.5, annot_kws={'size':11}, 
                          cbar_kws={'shrink':.8, 'label':'Mean upper tropospheric temperature ($^{\circ}$C)'})                       
            
        
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
        
        corrs = pd.DataFrame(list(zip(temp_series, rel_hum_series, mean_cover)), 
                             columns = ['T', 'RH', 'cirrus cover']).corr()
        
    def corr_meteo_cover(self, temp_list, rel_hum_list, calipso_list):
        
        nanmean = lambda x: np.nanmean(x)
        temp_series = list(map(nanmean, temp_list))
        rel_hum_series = list(map(nanmean, rel_hum_list))
        mean_cover = list(map(nanmean, calipso_list))
        
        corrs = pd.DataFrame(list(zip(temp_series, rel_hum_series, mean_cover)), 
                             columns = ['T', 'RH', 'cirrus cover']).corr()
        
        return corrs

    def spatial_cov_allmonths(self, axis):

        colors = [miscellaneous.random_color() for _ in range(100)]
        
        colors = random.sample(set(colors), len(months_complete))
        
        lon_pos_agg = []
        lat_pos_agg = []
        
        for month in months_complete:
            lidar = np.load(lidar_path + 'LIDAR_processed\\LIDAR_' + month + '.npy', 
                                        allow_pickle = True)
            lon_pos_agg.append(lidar.item()['lon_pos'])
            lat_pos_agg.append(lidar.item()['lat_pos'])
            
        array = lon_pos_agg if axis == 'Longitude' else lat_pos_agg
            
        plt.figure()
        
        for row in range(len(array)):
        
            # Draw the density plot
            sns.distplot(np.concatenate(array[row]), hist = False, kde = True,
                         kde_kws = {'linewidth': 3}, label = '{0}-{1}'.format(months_complete[row][0:2],
                                                                              months_complete[row][3:5]),
                                                                              color = colors[row])
            
        # Plot formatting
        plt.legend(prop={'size': 16}, ncol = 3, title = 'Month')
        plt.xlabel('{0} ($^{\circ}$)'.format(axis))
        plt.ylabel('Density')
