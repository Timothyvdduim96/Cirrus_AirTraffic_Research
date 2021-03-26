# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 13:07:04 2021

@author: T Y van der Duim
"""

'''
------------------------------PACKAGES-----------------------------------------
'''

import numpy as np
from SPACE_processing import L2_to_L3, bin_pressureL, hdf4_files, feature_class, convert_seconds, CALIPSO, dates, times
#from logistic_regmodel import netcdf_import
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from netCDF4 import Dataset
import datetime
from scipy import stats
import math as m


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


#%%
'''
------------------------TIME SERIES CALIPSO------------------------------------
'''

path = "E:\\Research_cirrus\\"

months = ['03_15', '06_15', '09_15', '12_15', '03_16', '06_16', '09_16', '12_16',
          '03_17', '06_17', '09_17', '12_17', '03_18', '06_18', '09_18', '12_18',
          '03_19', '06_19', '09_19', '12_19', '03_20', '06_20', '09_20', '12_20']

months_cover = []

calipso = CALIPSO(path + 'CALIPSO_data\\LIDAR_' + months[0], pressure_axis = False)

#def model_arrays(file, date_start, date_end, nr_periods, lag):
    
data = netcdf_import(path + "ERA5_data\\ERA5_03_15.nc")

#%%
rel_hum = np.mean(data['r'], axis = 1)
print(rel_hum.shape)
temp = np.mean(data['t'], axis = 1)

# select parts of ERA5 data at times when CALIPSO overpasses Europe
overpass_time = [datetime.datetime.combine(date, time) for date, time in zip(dates, times)] # merge dates and times to get a datetime list of Calipso overpasses

start = datetime.datetime.strptime("2015-03-01 00:00", '%Y-%m-%d %H:%M')
end = datetime.datetime.strptime("2015-03-31 23:00", '%Y-%m-%d %H:%M')

hour_list = pd.date_range(start = start, end = end, periods = 744).to_pydatetime().tolist()

lag = 2

idx = [key - lag for key, val in enumerate(hour_list) if val in overpass_time]
print(idx)
calipso = calipso[len(list(filter(lambda x: (x < 0), idx))):, :, :]
len(calipso)
idx = [item for item in idx if item >= 0]
rel_hum = rel_hum[idx, :, :]
print(rel_hum.shape)
temp = temp[idx, :, :]

def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]

remove_dup = dict((x, duplicates(overpass_time, x))[1] for x in set(overpass_time) 
                  if overpass_time.count(x) > 1)


if len(idx) != len(calipso):
    print("wrong")
    cirrus_cover = np.delete(calipso, list(remove_dup.values()), axis = 0)
else:
    cirrus_cover = calipso
    
# transpose 2nd and 3rd axes (lat and lon) so that it matches the cirrus cover array and match size
rel_hum = np.transpose(rel_hum, (0, 2, 1))[:, :-1, :-1] 
temp = np.transpose(temp, (0, 2, 1))[:, :-1, :-1]
rel_hum = np.where(np.isnan(cirrus_cover) == True, np.nan, rel_hum)
temp = np.where(np.isnan(cirrus_cover) == True, np.nan, temp)

# flatten the arrays (1D)
cirrus_cover = cirrus_cover.reshape(-1)
rel_hum = rel_hum.reshape(-1)
temp = temp.reshape(-1)

# remove nan instances
cirrus_cover = cirrus_cover[~(np.isnan(cirrus_cover))] 
rel_hum = rel_hum[~(np.isnan(rel_hum))] 
temp = temp[~(np.isnan(temp))]

binned_temp_cirrus = stats.binned_statistic_2d(temp, rel_hum,
                                     cirrus_cover, statistic = 'mean',
                                     bins = [10, 5],
                                     range = [[np.min(temp), np.max(temp)], [np.min(rel_hum), np.max(rel_hum)]])

binned_cirrus = binned_temp_cirrus.statistic
    #return 
#for month in months:

#     cirrus_cover = np.reshape(CALIPSO(path + 'LIDAR_' + month, pressure_axis = False), -1)
#     months_cover.append(cirrus_cover[~(np.isnan(cirrus_cover))])

# mean = lambda x: np.nanmean(x)

# mean_cover = list(map(mean, months_cover))

# dates = [month.replace('_', '-') for month in months]
# years = ['2015', '2016', '2017', '2018', '2019', '2020']

# cover_df = pd.DataFrame(list(zip(dates, mean_cover)), columns = ['dates', 'mean_cirrus_cover'])
# cover_df['dates'] = pd.to_datetime(cover_df['dates'], format='%m-%y')
# cover_df['month'] = cover_df['dates'].apply(lambda x: x.strftime("%m"))
# cover_df['year'] = cover_df['dates'].apply(lambda x: x.strftime("%Y"))
# cover_df = cover_df.drop(['dates'], axis = 1)
# #cover_df = cover_df.set_index('month', append=True)

# pivot_cirrus = pd.pivot_table(cover_df, values="mean_cirrus_cover",
#                                    index=["month"],
#                                    columns=["year"],
#                                    fill_value=0,
#                                    margins=True)

# pivot_cirrus.index = ['Mar', 'Jun', 'Sep', 'Dec', 'All']

# #%%

# ax = sns.heatmap(pivot_cirrus, cmap='seismic', robust=True, fmt='.2f', 
#                  annot=True, linewidths=.5, annot_kws={'size':11}, 
#                  cbar_kws={'shrink':.8, 'label':'Mean cirrus coverage'})                       
    
# ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)

# #%%
# plt.figure()
# plt.scatter(years, mean_cover[0::4], label = "March")
# plt.scatter(years, mean_cover[1::4], label = "June")
# plt.scatter(years, mean_cover[2::4], label = "September")
# plt.scatter(years, mean_cover[3::4], label = "December")
# plt.plot(years, [np.mean(mean_cover[0:4]), np.mean(mean_cover[4:8]),
#                  np.mean(mean_cover[8:12]), np.mean(mean_cover[12:16]),
#                  np.mean(mean_cover[16:20]), np.mean(mean_cover[20:24])])
# plt.legend()
# plt.show()

# #%%
# path = "E:\\Research_cirrus\\ERA5_data\\"

# P_lvls = np.arange(100, 450, 50)
# #test = netcdf_import(path + 'ERA5_15.nc', 250)

# ERA_years = ['ERA5_15.nc', 'ERA5_16.nc', 'ERA5_17.nc', 'ERA5_18.nc', 'ERA5_19.nc', 'ERA5_20.nc']

# T = []
# relhum = []

# for ERA_year in ERA_years:
#     print(ERA_year)
#     mar_t = []
#     jun_t = []
#     sep_t = []
#     dec_t = []
    
#     mar_r = []
#     jun_r = []
#     sep_r = []
#     dec_r = []
    
#     ERA_Data = netcdf_import(path + ERA_year)
#     ERA_temp = ERA_Data['t']
#     ERA_relhum = ERA_Data['r']
    
#     T.append(np.mean(ERA_temp[0:int(24 * 31)]))
#     T.append(np.mean(ERA_temp[int(24 * 31):int(24 * 31) + int(24 * 30)]))
#     T.append(np.mean(ERA_temp[int(24 * 31) + int(24 * 30):int(24 * 31) + int(24 * 30) + int(24 * 30)]))
#     T.append(np.mean(ERA_temp[int(24 * 31) + int(24 * 30) + int(24 * 30):]))
    
#     relhum.append(np.mean(ERA_relhum[0:int(24 * 31)]))
#     relhum.append(np.mean(ERA_relhum[int(24 * 31):int(24 * 31) + int(24 * 30)]))
#     relhum.append(np.mean(ERA_relhum[int(24 * 31) + int(24 * 30):int(24 * 31) + int(24 * 30) + int(24 * 30)]))
#     relhum.append(np.mean(ERA_relhum[int(24 * 31) + int(24 * 30) + int(24 * 30):]))

# print(T)

# print(relhum)   