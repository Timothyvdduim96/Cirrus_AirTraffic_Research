# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 13:49:08 2021

@author: fafri
"""
'''
------------------------------PACKAGES-----------------------------------------
'''

from netCDF4 import Dataset
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as m
from scipy import stats
import re
import random

'''
-------------------------------CLASS-------------------------------------------
'''

class miscellaneous:
    
    def random_color():
        return "#{}{}{}{}{}{}".format(*(random.choice("0123456789abcdef") for _ in range(6)))
    
    def dist(arr1, arr2):
        return abs(arr1 - arr2)
    
    def fl_to_km(fl):
        ft = fl * 100
        return ft * 3.048e-4
    
    def km_to_fl(km):
        fl = km / 3.048e-4
        return fl / 100
    
    def deg_to_km(deg):
        conv = 2 * m.pi * 6.378e3 / 360
        return deg * conv
    
    def duplicates(lst, item):
            return [i for i, x in enumerate(lst) if x == item]
    
    def remove_chars(row):
        try:
            row = re.sub('[ ,.!@#$-/:&]', '', row)
        except:
            row = row
        return row
    
    def hour_rounder(t):
            # Rounds to nearest hour by adding a timedelta hour if minute >= 30
            return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
                       + datetime.timedelta(hours=t.minute//30))
        
    def quarterhr_rounder(t):
            # Rounds to nearest hour by adding a timedelta hour if minute >= 30
            return t.replace(second=0, microsecond=0, minute=0,
                              hour=t.hour) + datetime.timedelta(minutes=round(t.minute / 15) * 15)
        
    def flatten_clean(array):
        reshaped = np.reshape(array, -1)
        return reshaped[~(np.isnan(reshaped))]
    
    def netcdf_import(file):
        data = Dataset(file,'r')
        my_dict = {}
        for key in data.variables.keys():
            if key == 'longitude' or key == 'latitude' or key == 'time' or key == 'level':
                my_dict[key] = data[key][:]
            else:
                my_dict[key] = data[key][:, :, :, :]
        return my_dict
    
    def multidim_slice(ind_arr, arr):
        output_arr = np.zeros((np.shape(arr)[1], np.shape(arr)[2]))
        for row in range(len(ind_arr)):
            for col in range(len(ind_arr[row])):
                ind = ind_arr[row, col]
                output_arr[row, col] = arr[ind, row, col]
        return output_arr
    
    def pivot_array(arr, dates):
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
    
    def dist2sat(df, current_lon, current_lat):
        
        def deg_to_km(deg):
            conv = 2 * m.pi * 6.378e3 / 360
            return deg * conv

        d_lon = deg_to_km(abs(df['Longitude'] - current_lon))
        d_lat = deg_to_km(abs(df['Latitude'] - current_lat))
        
        return np.sqrt(d_lon**2 + d_lat**2)
    
    def bin_2d(x_list, y_list, z_list, min_x, max_x, min_y, max_y, res_x, res_y, stat):
        grid = stats.binned_statistic_2d(np.array(x_list).flatten(),
                                                 np.array(y_list).flatten(),
                                                 np.array(z_list).flatten(),
                                                 statistic = stat,
                                                 bins = [int((max_x - min_x) / res_x), int((max_y - min_y) / res_y)],
                                                 range = [[min_x, max_x], [min_y, max_y]])
        
        return grid.statistic

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
               width = 0.2, alpha = 0.5, color = "green")
        
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=14)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks(np.arange(0, 2 * m.pi + 2 / 24 * m.pi, 2 / 24 * m.pi))
        ax.set_yticks(np.arange(0, 550, 100))
        ax.set_rlabel_position(83.5)
        #ax.set_title("Overpass frequency of CALIPSO over Europe during Mar, Jun,\n Sep & Dec from 2015 till 2020")
        ticks = [f"{i}:00" for i in range(0, 24, 1)]
        ax.set_xticklabels(ticks)
        
    