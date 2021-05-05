# -*- coding: utf-8 -*-
"""
@author: T Y van der Duim
"""

'''
------------------------------PACKAGES-----------------------------------------
'''

import numpy as np
from CALIPSO import CALIPSO_analysis
from miscellaneous import miscellaneous
from AIRCRAFT import flight_analysis, flights_after_2018
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import sys

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


class time_series:
    
    def exe(self, pressure_axis):
        
        self.calipso_day_list = []
        self.calipso_night_list = []
        self.calipso_total_list = []
        self.atd_list = []
        self.rel_hum_list = []
        self.temp_list = []
        self.calipso_supersat_list = []
        self.calipso_subsat_list = []
        self.issr_list = []
        self.calipso_supersat_list_night = []
        self.calipso_subsat_list_night = []
        self.calipso_supersat_list_day = []
        self.calipso_subsat_list_day = []
        
        for month in months_complete:
            self.CALIPSO(month, pressure_axis)
            self.air_traffic(month) if (month in months_1518) else ''
            self.meteo_data(month, pressure_axis)
            self.supersat_subsat_split()
            self.rel_hum_list.append(self.rel_hum_cleaned)
            self.temp_list.append(self.temp_cleaned)
            self.calipso_day_list.append(self.calipso_day_cleaned)
            self.calipso_night_list.append(self.calipso_night_cleaned)
            self.calipso_total_list.append(self.calipso_total_cleaned)
            self.calipso_supersat_list.append(self.calipso_supersat_cleaned)
            self.calipso_subsat_list.append(self.calipso_subsat_cleaned)
            self.calipso_supersat_list_night.append(self.calipso_supersat_cleaned_night)
            self.calipso_subsat_list_night.append(self.calipso_subsat_cleaned_night)
            self.calipso_supersat_list_day.append(self.calipso_supersat_cleaned_day)
            self.calipso_subsat_list_day.append(self.calipso_subsat_cleaned_day)
            self.issr_list.append(self.issr)
            self.atd_list.append(self.atd) if (month in months_1518) else ''
                 
    def CALIPSO(self, month, mode, pressure_axis):
                            
        lidar = CALIPSO_analysis(month, 15, pressure_axis)
        self.calipso = lidar.CALIPSO_cirrus
        self.dates = lidar.dates
        self.times = lidar.times
        self.lon_pos = lidar.lon_pos
        self.lat_pos = lidar.lat_pos
        
        savearray = {'calipso': self.calipso, 'dates': lidar.dates, 'times': lidar.times,
                      'lon_pos': lidar.lon_pos, 'lat_pos': lidar.lat_pos, 'opt_depth': lidar.cirrus_od}

        np.save('LIDAR_' + month, savearray, allow_pickle = True)          
            
        self.overpass_time = [datetime.datetime.combine(date, time) for date, time in 
                        zip(self.dates, self.times)]
        
        self.overpass_hit = [miscellaneous.hour_rounder(overpass) for overpass
                              in self.overpass_time]
        
        double_sets = np.where(np.diff(self.overpass_hit) == datetime.timedelta(0))[0]
        ind_keep = list(set(np.arange(len(self.overpass_time))).difference(double_sets))
        
        self.calipso_total = self.calipso[ind_keep, :, :]
        
        self.overpass_time_selec = [self.overpass_time[i] for i in ind_keep]

        nightpasses = pd.DataFrame(np.arange(0, len(self.overpass_time_selec)), index = self.overpass_time_selec,
                     columns = ['index']).between_time('22:00', '5:00')
        
        self.mask = np.zeros(len(self.calipso_total),dtype=bool)
        self.mask[nightpasses] = True
        self.calipso_night = self.calipso_total[self.mask]
        self.calipso_day = self.calipso_total[~self.mask]            
        
        self.calipso_total_cleaned = miscellaneous.flatten_clean(self.calipso)
        self.calipso_night_cleaned = miscellaneous.flatten_clean(self.calipso_night)
        self.calipso_day_cleaned = miscellaneous.flatten_clean(self.calipso_day)
        
    def air_traffic(self, month):
                                
        flight = flight_analysis(month)
        flight.individual_flights()
        flight.flighttracks()
        flight.merge_datasets()
        day_quarter_hours = pd.date_range(start = datetime.datetime.strptime('01-03-2015 00:00', '%d-%m-%Y %H:%M'), 
                                              end = datetime.datetime.strptime('31-03-2015 23:00', '%d-%m-%Y %H:%M'),
                                              freq = 'D').to_pydatetime().tolist()
        atd = flight.grid_ATD(flight.flights_merged, [6, 14], day_quarter_hours, 1440)
        self.atd = miscellaneous.flatten_clean(atd)
        np.save('Flights_{0}_allATD'.format(month), self.atd, allow_pickle = True)
                
    def meteo_data(self, month, pressure_axis): # ind_remove = np.where(np.diff(self.overpass_time) == datetime.timedelta(0))[0]
        
        ERA_Data = miscellaneous.netcdf_import(meteo_path + 'NewERA_5\\' + 'ERA5_' + month + '.nc')
        ERA_temp = ERA_Data['t']
        ERA_relhum = ERA_Data['r']

        hour_list = pd.date_range(start = miscellaneous.hour_rounder(self.overpass_time[0]), 
                                  end = miscellaneous.hour_rounder(self.overpass_time[-1]),
                                  freq = 'H').to_pydatetime().tolist()
        
        self.overpass_hit = [miscellaneous.hour_rounder(overpass) for overpass
                              in self.overpass_time]
        
        idx = [key for key, val in enumerate(hour_list) if val in self.overpass_hit]
        
        idx[-1] = idx[-1] - 1 if (idx[-1] >= len(ERA_relhum)) else idx[-1]

        rel_hum = ERA_relhum[idx, :, :, :]
        temp = ERA_temp[idx, :, :, :]
    
        remove_dup = dict((x, miscellaneous.duplicates(self.overpass_hit, x))[1] for x 
                          in set(self.overpass_hit) if self.overpass_hit.count(x) > 1)
        
        cirrus_cover = np.isnan(self.calipso)
        
        cirrus_cover = np.delete(cirrus_cover, list(remove_dup.values()), 
                                 axis = 0) if (len(idx) != len(cirrus_cover)) else cirrus_cover

        # reshape arrays to match format cirrus
        rel_hum = np.transpose(rel_hum, (0, 1, 3, 2))[:, :, :-1, :-1] 
        temp = np.transpose(temp, (0, 1, 3, 2))[:, :, :-1, :-1]
        
        if pressure_axis == False:
            # take mean in the vertical
            self.rel_hum = np.mean(rel_hum, axis = 1)
            self.temp = np.mean(temp, axis = 1)
        
        elif pressure_axis == True:
            rh_layer_list = []
            temp_layer_list = []
            
            # match locations to those where CALIPSO overpasses
            for i in range(11):
                rh_layer = rel_hum[:, i, :, :]
                temp_layer = temp[:, i, :, :]
                rh_layer_list.append(np.where(cirrus_cover == True, np.nan, rh_layer))
                temp_layer_list.append(np.where(cirrus_cover == True, np.nan, temp_layer))
            
            self.rel_hum = np.stack(rh_layer_list, axis = 1)
            self.temp = np.stack(temp_layer_list, axis = 1)
        
        self.rel_hum_cleaned = miscellaneous.flatten_clean(rel_hum)
        self.temp_cleaned = miscellaneous.flatten_clean(temp)
    
    def supersat_subsat_split(self):
        
        bool_supersub = np.any(self.rel_hum >= 100, axis = 1)
        calipso_supersat = np.where(bool_supersub == True, self.calipso_total, np.nan)
        calipso_subsat = np.where(bool_supersub == False, self.calipso_total, np.nan)
        self.calipso_supersat_cleaned = miscellaneous.flatten_clean(calipso_supersat)
        self.calipso_subsat_cleaned = miscellaneous.flatten_clean(calipso_subsat)
        
        bool_supersub_night = np.any(self.rel_hum[self.mask] >= 100, axis = 1)
        calipso_supersat_night = np.where(bool_supersub_night == True, self.calipso_night, np.nan)
        calipso_subsat_night = np.where(bool_supersub_night == False, self.calipso_night, np.nan)
        self.calipso_supersat_cleaned_night = miscellaneous.flatten_clean(calipso_supersat_night)
        self.calipso_subsat_cleaned_night = miscellaneous.flatten_clean(calipso_subsat_night)
        
        bool_supersub_day = np.any(self.rel_hum[~self.mask] >= 100, axis = 1)
        calipso_supersat_day = np.where(bool_supersub_day == True, self.calipso_day, np.nan)
        calipso_subsat_day = np.where(bool_supersub_day == False, self.calipso_day, np.nan)
        self.calipso_supersat_cleaned_day = miscellaneous.flatten_clean(calipso_supersat_day)
        self.calipso_subsat_cleaned_day = miscellaneous.flatten_clean(calipso_subsat_day)
        
        rel_hum_overpass = self.rel_hum[~np.isnan(self.rel_hum)]
        self.issr = len(rel_hum_overpass[rel_hum_overpass >= 100]) / len(rel_hum_overpass)
        
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
        ax = sns.heatmap(cirrus, cmap='seismic', robust=True, fmt='.3f', 
                          annot=True, linewidths=.5, annot_kws={'size':16}, 
                          cbar_kws={'shrink':.8, 'label':'cirrus cover'})                       
            
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=18)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=18)

    def dayvsnight_CALIPSO(self, calipso_daytime, calipso_nighttime):
        
        nanmean = lambda x: np.nanmean(x)
        calipso_nighttime = list(map(nanmean, calipso_nighttime))
        calipso_daytime = list(map(nanmean, calipso_daytime))
        
        calipso_difference = [(night - day) / day * 100 for day, night in zip(calipso_daytime, calipso_nighttime)]
        self.calipso_difference = miscellaneous.pivot_array(calipso_difference, dates)
        
        plt.figure()
        ax = sns.heatmap(self.calipso_difference, cmap='seismic', center = 0, robust=True, fmt='.1f', 
                          annot=True, linewidths=.5, annot_kws={'size':16}, 
                          cbar_kws={'shrink':.8, 'label':'$\Delta$ cirrus cover (%)'})                       
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=18)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=18)

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
                          annot=True, linewidths=.5, annot_kws={'size':18}, 
                          cbar_kws={'shrink':.8, 'label':'ATD (m km$^{-2}$ hr$^{-1}$)'})                       
            
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=18)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=18)
        
    def scatter_flights_vs_atd(self, atd_list):
        
        atdmeans = [np.mean(atd) for atd in atd_list]
        
        self.nr_flights = []

        for month in months_1518:
            print(month)
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
            
        self.X = np.array(self.nr_flights).reshape(-1, 1)
        
        atdmeans = [np.mean(atd) for atd in atd_list]
        reg = LinearRegression().fit(self.X, atdmeans)
        
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
        
        lm = sm.OLS(atdmeans, self.X).fit()
        
        fig, ax = plt.subplots(figsize=(12,8))
        sm.graphics.influence_plot(lm, ax= ax, criterion="cooks")
        
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
        
    def corr_meteo_cover(self, temp_list, rel_hum_list, calipso_list):
        
        nanmean = lambda x: np.nanmean(x)
        temp_series = list(map(nanmean, temp_list))
        rel_hum_series = list(map(nanmean, rel_hum_list))
        mean_cover = list(map(nanmean, calipso_list))
        
        corrs = pd.DataFrame(list(zip(temp_series, rel_hum_series, mean_cover)), 
                             columns = ['T', 'RH', 'cirrus cover']).corr()
        
        return corrs

    def spatial_cov_allmonths(self, axis):
        
        colors_greys = [[plt.cm.Greys(i / 8 + 0.25)] for i in range(6)]
        colors_purples = [[plt.cm.Blues(i / 8 + 0.25)] for i in range(6)]
        colors_greens = [[plt.cm.Greens(i / 8 + 0.25)] for i in range(6)]
        colors_reds = [[plt.cm.Reds(i / 8 + 0.25)] for i in range(6)]
        
        colors = []
        
        for year in range(len(years)):
            colors.append([colors_greys[year], colors_purples[year], colors_greens[year], 
                           colors_reds[year]])
            
        styles = ['solid', 'dashed', 'dotted', 'dashdot', (0, (5, 10)), (0, (3, 1, 1, 1, 1, 1))]
        styles = np.repeat(styles, 4)
        
        colors = [item for sublist in colors for item in sublist]
        
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
                                                                              color = colors[row][0])
            
        # Plot formatting
        plt.legend(prop={'size': 16}, ncol = 6, title = 'Month')
        plt.xlabel('position ($^{\circ}$)')
        plt.ylabel('density')

    def time_series_double(self):
        nanmean = lambda x: np.nanmean(x)
        calipso_nighttime = list(map(nanmean, self.calipso_night_list))
        calipso_daytime = list(map(nanmean, self.calipso_day_list))
        
        calipso_difference = [(night - day) / day * 100 for day, night in zip(calipso_daytime, calipso_nighttime)]
                
        cal = miscellaneous.pivot_array([np.mean(item) for item in self.calipso_total_list], dates)
        issr = miscellaneous.pivot_array(self.issr_list, dates)
        calipso_difference = miscellaneous.pivot_array(calipso_difference, dates)
        
        issr = np.round(calipso_difference, 1)
        df = issr.applymap(str)
        df = ' (' + df + '%)'
        
        df = np.round(cal, 3).applymap(str) + '\n' + df
        
        plt.figure()
        ax = sns.heatmap(cal, cmap='seismic',
                          annot=df, linewidths=.5, fmt = '',
                          cbar_kws={'shrink':.8, 'label':'cirrus cover'})                       
            
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=18)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=18)

    def super_vs_sub_daynight(self):   

        calipso_supersat_list_night = np.array(self.calipso_supersat_list_night)
        calipso_subsat_list_night = np.array(self.calipso_subsat_list_night)
        calipso_supersat_list_day = np.array(self.calipso_supersat_list_day)
        calipso_subsat_list_day = np.array(self.calipso_subsat_list_day)
        
        calipso_supersat_list_night = [np.mean(item) for item in calipso_supersat_list_night]
        calipso_subsat_list_night = [np.mean(item) for item in calipso_subsat_list_night]
        calipso_supersat_list_day = [np.mean(item) for item in calipso_supersat_list_day]
        calipso_subsat_list_day = [np.mean(item) for item in calipso_subsat_list_day]
        
        night = [(supersat - subsat) * 100 for supersat, subsat in zip(calipso_supersat_list_night, calipso_subsat_list_night)]
        day = [(supersat - subsat) * 100 for supersat, subsat in zip(calipso_supersat_list_day, calipso_subsat_list_day)]
        
        df = pd.DataFrame(list(zip(day, night)), columns = ['day', 'night'])
        df_melt = df.melt(value_vars = ['day',
                                        'night'],
                          var_name = '$\Delta cirrus (%) in supersaturated air vs sub-saturated air$')
        df_melt.columns = ['supersaturated air vs sub-saturated air', '$\Delta$ cirrus (%)']
        
        plt.figure()
        sns.boxplot(x = 'supersaturated air vs sub-saturated air', y = '$\Delta$ cirrus (%)', data = df_melt,
                    palette = 'Greens')
        sns.swarmplot(y='$\Delta$ cirrus (%)', x='supersaturated air vs sub-saturated air',
                      data=df_melt, 
                      color='black',
                      alpha=0.75)