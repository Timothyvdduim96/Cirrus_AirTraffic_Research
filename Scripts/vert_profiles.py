# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 15:22:43 2021

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
from scipy.interpolate import UnivariateSpline

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
----------------------------------CALIPSO-----------------------------------------
'''
all_calipso_2 = []

for month in months_complete[12:]:
    print(month)
    lidar = CALIPSO_analysis(lidar_path + 'LIDAR_' + month, 15, True)
    calipso = lidar.layered_cirrus
    calipso = np.vstack(calipso)
    all_calipso_2.append(calipso[np.isnan(calipso) == False])

all_cal = [all_calipso, all_calipso_2]
all_calipso.extend(all_calipso_2)
del all_calipso[12]
# mar_calipso = np.concatenate(all_calipso[0::4])
# jun_calipso = np.concatenate(all_calipso[1::4])
# sep_calipso = np.concatenate(all_calipso[2::4])
# dec_calipso = np.concatenate(all_calipso[3::4])
# tot_calipso = [mar_calipso, jun_calipso, sep_calipso, dec_calipso]

#%%

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
fig = plt.figure()
ax1 = fig.add_subplot(111)

idx = 0
for calipso in all_calipso:
    sns.kdeplot(calipso, vertical = True,label = '{0}-{1}'.format(months_complete[idx][0:2],
                                                                  months_complete[idx][3:5]),
                                                                  color = colors[idx][0], ax = ax1)
    idx += 1
    
#plt.ylim([np.min(all_calipso), np.max(all_calipso)])
ax1.set_yscale('log')
ax1.set_ylim(10 * np.ceil(np.max([arr.max() for arr in all_calipso]) / 10),
             np.min([arr.min() for arr in all_calipso])) # avoid truncation of 1000 hPa
subs = [1,2,5]

if np.max([arr.max() for arr in all_calipso]) / np.min([arr.min() for arr in all_calipso]) < 30:
    subs = [1,2,3,4,5,6,7,8,9]
    
y1loc = matplotlib.ticker.LogLocator(base=10, subs=subs)
ax1.yaxis.set_major_locator(y1loc)
ax1.set_xlabel("Cirrus cover KDE")
ax1.set_ylabel("Pressure [hPa]")
fmt = matplotlib.ticker.FormatStrFormatter("%g")
ax1.yaxis.set_major_formatter(fmt)
#plt.gca().invert_yaxis()
z0 = 8.333    # scale height for pressure_to_altitude conversion [km]
altitude = [z0 * np.log(1013.25 / arr) for arr in all_calipso]
# add second y axis for altitude scale 
axr = ax1.twinx()
label_xcoor = 1.05
axr.set_ylabel("Altitude [km]")
axr.yaxis.set_label_coords(label_xcoor, 0.5)
axr.set_ylim(np.min([arr.min() for arr in altitude]), np.max([arr.max() for arr in altitude]))
yrloc = matplotlib.ticker.MaxNLocator(steps=[1,2,5,10])
axr.yaxis.set_major_locator(yrloc)
axr.yaxis.tick_right()
axr.grid(False)
legend = ax1.legend(prop={'size': 14}, ncol = 6, title = 'Month', frameon = True)
frame = legend.get_frame()
frame.set_color('white')
plt.show()

#%%
'''
----------------------------------METEO-----------------------------------------
'''
ERA_temp_vert = []
ERA_relhum_vert = []

for month in months_complete:
    print(month)
    ERA_Data = miscellaneous.netcdf_import(meteo_path + 'ERA5_' + month + '.nc')
    ERA_temp = ERA_Data['t']
    ERA_relhum = ERA_Data['r']
    print(np.shape(ERA_temp))
    ERA_temp_vert.append(np.mean(ERA_temp, axis = (0, 2, 3)))
    ERA_relhum_vert.append(np.mean(ERA_relhum, axis = (0, 2, 3)))

mar_temp = np.mean(np.vstack(ERA_temp_vert)[0::4, :], axis = 0)
jun_temp = np.mean(np.vstack(ERA_temp_vert)[1::4, :], axis = 0)
sep_temp = np.mean(np.vstack(ERA_temp_vert)[2::4, :], axis = 0)
dec_temp = np.mean(np.vstack(ERA_temp_vert)[3::4, :], axis = 0)
tot_temp = [mar_temp, jun_temp, sep_temp, dec_temp]

mar_RH = np.mean(np.vstack(ERA_relhum_vert)[0::4, :], axis = 0)
jun_RH = np.mean(np.vstack(ERA_relhum_vert)[1::4, :], axis = 0)
sep_RH = np.mean(np.vstack(ERA_relhum_vert)[2::4, :], axis = 0)
dec_RH = np.mean(np.vstack(ERA_relhum_vert)[3::4, :], axis = 0)
tot_RH = [mar_RH, jun_RH, sep_RH, dec_RH]
#%%
pressures = [100, 150, 200, 250, 300, 350, 400]
from scipy.interpolate import CubicSpline
from scipy import optimize


def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])


def vert_profile_plot(var, interp, meteo):
    
    fig, ax1 = plt.subplots()

    var_min = 100#np.min([arr.min() for arr in tot_calipso])
    var_max = 400#np.max([arr.max() for arr in tot_calipso])
    
    idx = 0
    
    for month in var:
        #pressure = [pressures[i] for i in np.argsort(month).tolist()]
        #spl = CubicSpline(pressures, month)
        
        if interp == 'linear':
            tck = interpolate.splrep(pressures, month, k=2, s=0)
            xs = np.linspace(min(pressure), max(pressure), 100)
            sns.lineplot(interpolate.splev(xs, tck, der=0), xs, sort = False, 
                         label = '{0}-{1}'.format(months_complete[idx][0:2],
                         months_complete[idx][3:5]), color = colors[idx][0])
        
        elif interp == 'cubic':
            spl = CubicSpline(pressures, month, bc_type = 'natural')
            xs = np.linspace(min(pressure), max(pressure), 50)
            sns.lineplot(spl(xs), xs, sort = False,
                         label = '{0}-{1}'.format(months_complete[idx][0:2],
                         months_complete[idx][3:5]), color = colors[idx][0])
        
        idx += 1
        
    plt.ylim([var_min, var_max])
    ax1.set_yscale('log')
    ax1.set_ylim(10 * np.ceil(var_max / 10), var_min) # avoid truncation of 1000 hPa
    subs = [1,2,5]
    
    if var_max / var_min < 30:
        subs = [1,2,3,4,5,6,7,8,9]
        
    if meteo == 'temp':
        ax1.set_xlabel("Temperature [K]")
    if meteo == 'RH':
        ax1.set_xlabel("Relative Humidity [%]")
    y1loc = matplotlib.ticker.LogLocator(base=10, subs=subs)
    ax1.yaxis.set_major_locator(y1loc)
    ax1.set_ylabel("Pressure [hPa]")
    fmt = matplotlib.ticker.FormatStrFormatter("%g")
    ax1.yaxis.set_major_formatter(fmt)
    #plt.gca().invert_yaxis()
    z0 = 8.333    # scale height for pressure_to_altitude conversion [km]
    altitude = [z0 * np.log(1013.25 / arr) for arr in pressures]
    # add second y axis for altitude scale 
    axr = ax1.twinx()
    label_xcoor = 1.05
    axr.set_ylabel("Altitude [km]")
    axr.yaxis.set_label_coords(label_xcoor, 0.5)
    axr.set_ylim(np.min([arr.min() for arr in altitude]), np.max([arr.max() for arr in altitude]))
    yrloc = matplotlib.ticker.MaxNLocator(steps=[1,2,5,10])
    axr.yaxis.set_major_locator(yrloc)
    axr.yaxis.tick_right()
    axr.yaxis.set_major_locator(yrloc)
    axr.yaxis.tick_right()
    axr.grid(False)
    legend = ax1.legend(prop = {'size': 14}, ncol = 6, title = 'Month', frameon = True)
    frame = legend.get_frame()
    frame.set_color('white')
    plt.show()
    
vert_profile_plot(ERA_temp_vert, 'linear', 'temp')
vert_profile_plot(ERA_relhum_vert, 'cubic', 'RH')