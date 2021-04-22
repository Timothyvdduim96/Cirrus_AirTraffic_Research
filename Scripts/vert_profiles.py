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
from scipy.interpolate import CubicSpline
from scipy import optimize
from scipy import interpolate

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
all_calipso = []

for month in months_complete:
    print(month)
    lidar = CALIPSO_analysis(lidar_path + 'LIDAR_' + month, 15, True)
    calipso = lidar.layered_cirrus
    calipso = np.vstack(calipso)
    all_calipso_2.append(calipso[np.isnan(calipso) == False])
    
#%%
months_vert_ATD = []

for month in months_1518:
    print(month)
    flight_selec = np.load(flight_path + 'Flights_' + str(month) + '.pkl', allow_pickle = True)
    max_time = max(flight_selec['Time Over'])
    timedelta_sec = (max(flight_selec['Time Over']) - min(flight_selec['Time Over'])).seconds
    timedelta_min = int(timedelta_sec / 60)
    
    height = 6
    high_res = 0.5
    month_vert_ATD = []
    max_time = max_time.to_pydatetime()
        
    while height <= 13.5:
        print(height)
        flights = flight_analysis(str(month))
        flights.grid_ATD(flight_selec, [height, height + high_res], [max_time], timedelta_min)
        mean_flights_atd = np.mean(flights.flights_ATD)
        print(mean_flights_atd)
        month_vert_ATD.append(mean_flights_atd)
        height += high_res
    
    months_vert_ATD.append(month_vert_ATD)
#%%

all_calipso = np.load('C:\\Users\\fafri\\Documents\\ads_cirrus_airtraffic\\Scripts\\vert_profile_cirrus.npy',
                      allow_pickle = True)

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

#%%
fig = plt.figure()
ax1 = fig.add_subplot(111)

idx = 0
for calipso in all_calipso:
    sns.kdeplot(calipso, vertical = True, label = '{0}-{1}'.format(months_complete[idx][0:2],
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
ax1.set_xlabel("cirrus cover KDE")
ax1.set_ylabel("Pressure (hPa)")
fmt = matplotlib.ticker.FormatStrFormatter("%g")
ax1.yaxis.set_major_formatter(fmt)
#plt.gca().invert_yaxis()

def alt(h_b, P_ref, T_ref, P):
        
        M = 0.0289644 # molar mass Earths air in kg/mol
        R = 8.3144598 # universal gas constant in J/(mol K)
        g_0 = 9.80665 # gravitational constant
        
        return (R * T_ref / (g_0 * M) * np.log(P_ref / P) + h_b) / 1e3
    
altitude = []

pressure_list = np.arange(np.min([arr.min() for arr in all_calipso]),
             10 * np.ceil(np.max([arr.max() for arr in all_calipso]) / 10), 50)
    
for arr in pressure_list:
    P = alt(0, 1013.25, 288, arr) if alt(0, 1013.25, 288, arr) < 11 else alt(11000, 226.321, 216, arr)
    altitude.append(P)
        
# add second y axis for altitude scale 
axr = ax1.twinx()
label_xcoor = 1.05
axr.set_ylabel("Altitude (km)")
axr.yaxis.set_label_coords(label_xcoor, 0.5)
axr.set_ylim(np.min([arr.min() for arr in altitude]), np.max([arr.max() for arr in altitude]))
yrloc = matplotlib.ticker.MaxNLocator(steps=[1,2,5,10])
axr.yaxis.set_major_locator(yrloc)
axr.yaxis.tick_right()
axr.grid(False)
legend = ax1.legend(prop={'size': 16}, ncol = 6, title = 'Month', frameon = True)
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
    ERA_Data = miscellaneous.netcdf_import(meteo_path + 'NewERA_5\\' + 'ERA5_' + month + '.nc')
    ERA_temp = ERA_Data['t']
    ERA_relhum = ERA_Data['r']
    print(np.shape(ERA_temp))
    ERA_temp_vert.append(np.mean(ERA_temp, axis = (0, 2, 3)))
    ERA_relhum_vert.append(np.mean(ERA_relhum, axis = (0, 2, 3)))

#%%

def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

#vert_prof_ATD = vert_ATD.vert_prol_ATD

def vert_profile_plot(var, interp, meteo):
    
    fig, ax1 = plt.subplots()
    
    pressures = 101325 * np.exp(-0.00012 * np.arange(6250, 14250, 500)) / 100
    print(pressures)
    
    var_min = 100#np.min([arr.min() for arr in tot_calipso])
    var_max = 450#np.max([arr.max() for arr in tot_calipso])
    
    if meteo == 'ATD' or meteo == 'ATD_months':
        var_min = pressures.min()#np.min([arr.min() for arr in tot_calipso])
        var_max = pressures.max()#np.max([arr.max() for arr in tot_calipso])
    idx = 0
    
    for month in var:
        
        if interp == 'linear':
            pressures = [100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450]
            tck = interpolate.splrep(pressures, month, k=2, s=0)
            xs = np.linspace(min(pressures), max(pressures), 100)
            sns.lineplot(interpolate.splev(xs, tck, der=0), xs, sort = False, 
                         label = '{0}-{1}'.format(months_complete[idx][0:2],
                         months_complete[idx][3:5]), color = colors[idx][0])
            
            idx += 1
        
        elif interp == 'cubic':
            pressures = [100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450]
            spl = CubicSpline(pressures, month, bc_type = 'natural')
            xs = np.linspace(min(pressures), max(pressures), 50)
            sns.lineplot(spl(xs), xs, sort = False,
                         label = '{0}-{1}'.format(months_complete[idx][0:2],
                         months_complete[idx][3:5]), color = colors[idx][0])
            
            idx += 1
            
        else:
            pass
    
    if meteo == 'ATD':
        args = np.argsort(pressures)
        vert_prof_ATD = [var[arg] for arg in args]
        pressures = np.sort(pressures)
        spl = CubicSpline(pressures, vert_prof_ATD, bc_type = 'natural')
        xs = np.linspace(min(pressures), max(pressures), 40)
        sns.lineplot(spl(xs), xs, sort = False,
                     label = '{0}-{1}'.format(months_complete[idx][0:2],
                     months_complete[idx][3:5]))

        
    elif meteo == 'ATD_months':
        idx = 0
        for month in var:
            pressures = 101325 * np.exp(-0.00012 * np.arange(6250, 14250, 500)) / 100
            args = np.argsort(pressures)
            month = [month[arg] * 1000 for arg in args]
            pressures = np.sort(pressures)
            spl = CubicSpline(pressures, month, bc_type = 'natural')
            xs = np.linspace(min(pressures), max(pressures), 100)
            sns.lineplot(spl(xs), xs, sort = False,
                          label = '{0}-{1}'.format(months_complete[idx][0:2],
                          months_complete[idx][3:5]), color = colors[idx][0])
            # sns.lineplot(month, pressures, sort = False,
            #           label = '{0}-{1}'.format(months_complete[idx][0:2],
            #           months_complete[idx][3:5]), color = colors[idx][0])
            idx += 1

        
    plt.ylim([var_min, var_max])
    ax1.set_xlabel("ATD (m km$^{-2}$ hr$^{-1}$)")
    plt.tick_params(axis='y', which='minor')
    ax1.set_yscale('log')
    ax1.set_ylim(10 * np.ceil(var_max / 10), var_min) # avoid truncation of 1000 hPa
    subs = (0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )
    
    if var_max / var_min < 30:
        subs = [1,2,3,4,5,6,7,8,9]
        
    if meteo == 'temp':
        ax1.set_xlabel("temperature (K)")
    elif meteo == 'RH':
        ax1.set_xlabel("relative humidity (%)")
    elif meteo == 'ATD':
        ax1.set_xlabel("ATD (km km$^{-2}$ hr$^{-1}$)")
        
    ax1.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(1, 10.1, 0.5)))
    ax1.set_ylabel("pressure (hPa)")
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))

    #plt.gca().invert_yaxis()
    z0 = 8.333    # scale height for pressure_to_altitude conversion [km]
    
    def alt(h_b, P_ref, T_ref, P):
        
        M = 0.0289644 # molar mass Earths air in kg/mol
        R = 8.3144598 # universal gas constant in J/(mol K)
        g_0 = 9.80665 # gravitational constant
        
        return (R * T_ref / (g_0 * M) * np.log(P_ref / P) + h_b) / 1e3
    
    altitude = []
    
    for arr in pressures:
        P = alt(0, 1013.25, 288, arr) if alt(0, 1013.25, 288, arr) < 11 else alt(11000, 226.321, 216, arr)
        altitude.append(P)
        
    # add second y axis for altitude scale 
    axr = ax1.twinx()
    label_xcoor = 1.05
    axr.set_ylabel("Altitude (km)")
    axr.yaxis.set_label_coords(label_xcoor, 0.5)
    axr.set_ylim(np.min(altitude), np.max(altitude))
    yrloc = matplotlib.ticker.MaxNLocator(steps=[1,2,5,10])
    axr.yaxis.set_major_locator(yrloc)
    axr.yaxis.tick_right()
    axr.yaxis.set_major_locator(yrloc)
    axr.yaxis.tick_right()
    axr.grid(False)
    if meteo == 'ATD_months':
        legend = ax1.legend(prop = {'size': 16}, ncol = 4, title = 'Month', frameon = True)
    else:
        legend = ax1.legend(prop = {'size': 16}, ncol = 6, title = 'Month', frameon = True)
    frame = legend.get_frame()
    frame.set_color('white')
    plt.show()
    
#vert_profile_plot(ERA_temp_vert, 'linear', 'temp')
#vert_profile_plot(ERA_relhum_vert, 'cubic', 'RH')
test = vert_prol_ATD_list[0:2]
vert_profile_plot(vert_prol_ATD_list, '', 'ATD_months')
#%%

class flights_vert_res:
    
    def __init__(self, start_date, end_date, month):
        #self.flights = flight_analysis('03_15')
        #self.flights.exe_ac('load', 'None', 'None',
        #           'None','None', 'None','None',
        #           'None', 'None', '_full')
        self.height = 6
        self.high_res = 0.5
        self.flights = flight_analysis(month)
        self.flights.individual_flights()
        self.flights.flighttracks()
        self.flights.merge_datasets_no_interpol()
        #flight.exe_ac(mode = mode, resample_interval = '1min', window = 30, V_aircraft = 1200,
        #                    dates = self.dates, times = self.times, lon_pos = self.lon_pos,
        #                    lat_pos = self.lat_pos, savename = 'interpol_03_19')
        #flight.grid_ATD(flight.combined_datetime, 30)
        #atd = flight.flights_ATD
        #atd = np.where(np.isnan(self.calipso) == True, np.nan, atd)
        self.quarter_hours = pd.date_range(start = datetime.strptime(start_date, '%d-%m-%Y %H:%M'), 
                                  end = datetime.strptime(end_date, '%d-%m-%Y %H:%M'),
                                  freq = '15min').to_pydatetime().tolist()
        self.quarter_hours_day = pd.DataFrame(np.arange(0, len(self.quarter_hours)), 
                                              index = self.quarter_hours, 
                                              columns = ['index']).between_time('07:00', '18:00')
        
        index = self.quarter_hours_day['index'].tolist()
        
        self.quarter_hours_day = [self.quarter_hours[idx] for idx in index]
        self.vert_ATD()  
    
    def vert_ATD(self):
        
        self.vert_prol_ATD = []
        
        while self.height <= 13.5:
            print(self.height)
            self.flights.grid_ATD(self.flights.flights_merged, [self.height, self.height + self.high_res], self.quarter_hours_day, 15)
            mean_flights_atd = np.mean(self.flights.flights_ATD)
            print(mean_flights_atd)
            self.vert_prol_ATD.append(mean_flights_atd)
            self.height += self.high_res
            
#vert_prof_ATD = vert_ATD.vert_prol_ATD
#vert_profile_plot(vert_prof_ATD, '', 'ATD')

#%%
# heights = np.arange(6.25, 14.25, 0.5)
# P = 101325 * np.exp(-0.12 * heights)
# plt.figure()
# plt.scatter(vert_prof_ATD, P)

vert_prol_ATD_list.extend(vert_prol_ATD_list3)

vert_prol_ATD_list3 = []
start_dates = ['01-03-2015 00:00', '01-06-2015 00:00', '01-09-2015 00:00', '01-12-2015 00:00',
               '01-03-2016 00:00', '01-06-2016 00:00', '01-09-2016 00:00', '01-12-2016 00:00',
               '01-03-2017 00:00', '01-06-2017 00:00', '01-09-2017 00:00', '01-12-2017 00:00',
               '01-03-2018 00:00', '01-06-2018 00:00', '01-09-2018 00:00', '01-12-2018 00:00']
end_dates = ['15-03-2015 23:45', '15-06-2015 23:45', '15-09-2015 23:45', '15-12-2015 23:45',
               '15-03-2016 23:45', '15-06-2016 23:45', '15-09-2016 23:45', '15-12-2016 23:45',
               '15-03-2017 23:45', '15-06-2017 23:45', '15-09-2017 23:45', '15-12-2017 23:45',
               '05-03-2018 23:45', '05-06-2018 23:45', '05-09-2018 23:45', '05-12-2018 23:45']

idx = 12
for month in months_1518[12:]:
    print(month)
    flights = flights_vert_res(start_dates[idx], end_dates[idx], month)
    vert_prol_ATD_list3.append(flights.vert_prol_ATD)
    idx += 1
