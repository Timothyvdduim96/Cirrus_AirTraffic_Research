# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 18:16:01 2020

@author: T Y van der Duim
"""

'''
------------------------------PACKAGES-----------------------------------------
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ['PROJ_LIB'] = r"C:\Users\fafri\anaconda3\pkgs\proj4-6.1.1-hc2d0af5_1\Library\share"
from mpl_toolkits.basemap import Basemap
import numpy as np
from datetime import date, timedelta, datetime
import matplotlib.animation as animation
from netCDF4 import Dataset
from pyhdf.SD import SD, SDC
import json
from scipy import stats
import math as m
import urllib
from bs4 import BeautifulSoup
import re

path = "E:\\Research_cirrus\\"
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

#%%
'''
-------------------------------INDIVIDUAL FLIGHTS------------------------------
'''

'''
-------------------------------import and format------------------------------
'''

# import flights
flights_0315 = pd.read_csv(path + "Flight_data\\Flights_20150301_20150331.csv", sep=";")

# rename columns
flights_0315.columns = ["ID", "ICAO_dep", "lat_dep", "lon_dep", "ICAO_dest", "lat_dest", "lon_dest",
                        "planned_dep_time", "planned_arr_time", "dep_time", "arr_time", "AC_type",
                        "AC_operator", "AC_regis", "flight_type", "market", "req_FL", "dist"]

print("Original dataset consists of {0} flights.".format(len(flights_0315)))

'''
-----------------------------------filtering-----------------------------------
'''

# invalid values
crucial_cols = ["ID", "ICAO_dep", "lat_dep", "lon_dep", "ICAO_dest", "lat_dest", "lon_dest",
                        "dep_time", "arr_time", "AC_type", "AC_operator", "flight_type"] # columns that should not contain NaN
nan_flights_0315 = flights_0315[flights_0315[crucial_cols].isna().any(axis=1)]

print("Number of flights with important missing data is {0}.".format(len(nan_flights_0315)))

unknown_dep_airports = nan_flights_0315[nan_flights_0315["lat_dep"].isna()]["ICAO_dep"].unique()
unknown_dest_airports = nan_flights_0315[nan_flights_0315["lat_dest"].isna()]["ICAO_dest"].unique()
unknown_airports = list(set(unknown_dep_airports) | set(unknown_dest_airports))

# recover locations of known airports (ZZZZ and AFIL are not locatable!)
flights_0315.loc[flights_0315["ICAO_dep"] == "FAOR", "lat_dep"] = -26
flights_0315.loc[flights_0315["ICAO_dep"] == "FAOR", "lon_dep"] = 28
flights_0315.loc[flights_0315["ICAO_dest"] == "FAOR", "lat_dest"] = -26
flights_0315.loc[flights_0315["ICAO_dest"] == "FAOR", "lon_dest"] = 28

flights_0315.loc[flights_0315["ICAO_dep"] == "GQNN", "lat_dep"] = 18
flights_0315.loc[flights_0315["ICAO_dep"] == "GQNN", "lon_dep"] = -15
flights_0315.loc[flights_0315["ICAO_dest"] == "GQNN", "lat_dest"] = 18
flights_0315.loc[flights_0315["ICAO_dest"] == "GQNN", "lon_dest"] = -15

nan_flights_0315 = flights_0315[flights_0315[crucial_cols].isna().any(axis=1)]

print("Number of flights with important missing data after recovering known airport locations is {0}.".format(len(nan_flights_0315)))

flights_0315 = flights_0315[~flights_0315[crucial_cols].isna().any(axis=1)]

print("Number of flights after filtering those rows is {0}.".format(len(flights_0315)))

# not so interesting columns that can be dropped
droplabels = ["lat_dep", "lon_dep", "lat_dest", "lon_dest", "planned_dep_time", "planned_arr_time",
              "dep_time", "arr_time", "req_FL"]

flights_0315 = flights_0315.drop(labels = droplabels, axis = 1)

#%%
airports = pd.read_csv(path + "Additional_data\\airports.csv", sep=",", encoding='latin-1')

airports = airports[["airport", "ICAO"]]

# change colname to merge on departure airport
airports.columns = ["airport", "ICAO_dep"]
flights_0315 = flights_0315.merge(airports, how = "inner", on = ["ICAO_dep"])

# change colname to merge on arrival airport
airports.columns = ["airport", "ICAO_dest"]
flights_0315 = flights_0315.merge(airports, how = "inner", on = ["ICAO_dest"])

flights_0315.columns = ["ID", "ICAO_dep", "ICAO_dest", "AC_type", "AC_operator",
                        "AC_regis", "flight_type", "market", "dist", "dep_airport", "dest_airport"]

airport_count = flights_0315.groupby(by = "dep_airport")
airport_count = airport_count.size().rename("count")
airport_count = airport_count.sort_values(ascending = False)

fig, ax = plt.subplots()
ax.tick_params(axis='y', labelsize = 10, rotation = 30)
ax.tick_params(axis='x', labelsize = 14, rotation = 0)
ax.set_title("Number of flights departing in March 2015")
ax.barh(airport_count.index[0:20], airport_count[0:20])
ax.set_xticks(np.arange(0, 22000, 2000))
ax.set_facecolor('darkgrey')


#%%
'''
-------------------------------RADAR FLIGHTS-----------------------------------
'''

res_lon = 0.25 # set desired resolution in longitudinal direction in degs
res_lat = 0.25 # set desired resolution in latitudinal direction in degs
min_lon = -10 # research area min longitude
max_lon = 40 # research area max longitude
min_lat = 35 # research area min latitude
max_lat = 60 # research area max latitude

flighttrack_0315 = pd.read_csv(path + "Flight_data\\Flight_Points_Actual_20150301_20150331.csv", sep=",")

print("Original dataset consists of {0} flightpoints.".format(len(flighttrack_0315)))

# check for NaN values
nan_flightpoint_0315 = flighttrack_0315[flighttrack_0315.isna().any(axis=1)]

# all NaN values are in cols Latitude & Longitude -> drop
flighttrack_0315 = flighttrack_0315.dropna(how = 'any')

print("Number of flight points after dropping NaN locations is {0}.".format(len(flighttrack_0315)))

# only keep points within ROI
flighttrack_0315 = flighttrack_0315[(flighttrack_0315["Latitude"] > min_lat) & 
                                    (flighttrack_0315["Latitude"] < max_lat) & 
                                    (flighttrack_0315["Longitude"] > min_lon) & 
                                    (flighttrack_0315["Longitude"] < max_lon)]

print("Number of flight points after selecting ROI is {0}.".format(len(flighttrack_0315)))

# only keep data above 6 km (where cirrus can form)
flighttrack_0315 = flighttrack_0315[flighttrack_0315["Flight Level"] > 197]

print("Number of flight points after omitting points below 6 km is {0}.".format(len(flighttrack_0315)))

flighttrack_0315['Time Over'] = pd.to_datetime(flighttrack_0315['Time Over'], format="%d-%m-%Y %H:%M:%S")

#%%
'''
-----------------------map movements between two dates-------------------------
'''

start_date = '01-03-2015 12:00:00'
start_date = datetime.strptime(start_date, '%d-%m-%Y %H:%M:%S')
end_date = '01-03-2015 15:00:00'
end_date = datetime.strptime(end_date, '%d-%m-%Y %H:%M:%S')

filtered_flights = flighttrack_0315[flighttrack_0315['Time Over'].between(start_date, end_date)]

flight_IDs = flighttrack_0315["ECTRL ID"].unique()

fig = plt.figure(figsize=(16,12))
ax = plt.subplot(111)
plt.title("All registered flights (after filtering) on 3 March 2015 between 12PM and 3PM")
plt.rcParams.update({'font.size': 16})
map=Basemap(projection="lcc",resolution="i",width=5E6,height=2.5E6,
                              lon_0=15,lat_0=47.5,fix_aspect=False)
map.drawcoastlines()
map.drawcountries()
map.bluemarble()

#create and draw meridians and parallels grid lines
map.drawparallels(np.arange( -90., 120.,10.),labels=[1,0,0,0],fontsize=10)
map.drawmeridians(np.arange(-180.,180.,10.),labels=[0,0,0,1],fontsize=10)

for ID in filtered_flights["ECTRL ID"].unique():
    #convert latitude/longitude values to plot x/y values
    x, y = map(np.array(filtered_flights[filtered_flights['ECTRL ID'] == ID]['Longitude']),
              np.array(filtered_flights[filtered_flights['ECTRL ID'] == ID]['Latitude']))
    map.plot(x, y, linewidth=0.5, alpha = 0.3)

#%%
'''
--------------------INTERPOLATION TO 15 MIN INTERVALS--------------------------
'''
# def interpol(ID):
#     data = flighttrack_0315[flighttrack_0315['ECTRL ID'] == ID][['Time Over', 'Flight Level', 'Longitude', 'Latitude']]
#     interp = data.set_index('Time Over').resample('5min').mean().interpolate(method='linear')
#     interp = interp.reset_index()
#     interp['ID'] = [ID] * len(interp)
#     return interp

# cols = ['Time Over', 'Flight Level', 'Longitude', 'Latitude',  'ID']
# flight_0315_interp = pd.DataFrame(columns=cols)
# idx = 0

# for flight in flighttrack_0315['ECTRL ID'].unique():
#     data = interpol(flight)
#     flight_0315_interp = flight_0315_interp.append(data, ignore_index = True)
#     idx += 1
#     print(idx)
    
# flight_0315_interp.to_pickle('interpolated_flightdata_0315.pkl')

flight_0315_interp = pd.read_pickle(path + "Flight_data\\interpolated_flightdata_0315.pkl") #to load 123.pkl back to the dataframe df

#%%
'''
-------------------------------MERGE DATASETS----------------------------------
'''

flights = flight_0315_interp.merge(flights_0315, how = "inner", on = ["ID"])

#%%
'''
-------------------------------ANIMATE FLIGHTS--------------------------------
'''
# Set up formatting for the movie files
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])

fig = plt.figure(figsize=(16,12))
ax = plt.subplot(111)
plt.rcParams.update({'font.size': 16})
map=Basemap(projection="lcc",resolution="i",width=4E6,height=4E6,
                              lon_0=9.9167,lat_0=51.5167,fix_aspect=False)
map.drawcountries(color="black", linewidth=1)
map.shadedrelief()

#create and draw meridians and parallels grid lines
map.drawparallels(np.arange( -90., 120.,30.),labels=[1,0,0,0],fontsize=10)
map.drawmeridians(np.arange(-180.,180.,30.),labels=[0,0,0,1],fontsize=10)

nr_hours = 3

x, y = map(0, 0)
point = map.plot(x, y, 'ro', markersize = 3, color = 'red')[0]

colors = {'Traditional Scheduled':'blue', 'All-Cargo':'orange', 'Lowcost':'green',
          'Charter':'red', 'Business Aviation':'purple'}

text = plt.text(0.5, 1.05, str(start_date), ha="center", transform=ax.transAxes,)

def animate(i):
    #for hr in range(nr_hours):
    global text
    text.remove()
    start_date = '01-03-2015 00:00:00'
    start_date = datetime.strptime(start_date, '%d-%m-%Y %H:%M:%S') + timedelta(minutes = 5 * i)
    end_date = start_date + timedelta(minutes = 5)
    filtered_flights = flights[flights['Time Over'].between(start_date, end_date)]
    x, y = map(np.array(filtered_flights['Longitude']), np.array(filtered_flights['Latitude']))
    point.set_data(x, y)
    text = ax.text(0.5, 1.05, str(start_date), ha="center", transform=ax.transAxes,)
    return point,

myAnimation = animation.FuncAnimation(fig, animate, frames=300, interval=200)
#myAnimation.save('aircraft.mp4', writer=writer)

#%%
'''
---------------------------GRID AIR TRAFFIC (ATD)------------------------------
'''
def dist(arr1, arr2):
    return abs(arr1 - arr2)

def fl_to_km(fl):
    ft = fl * 100
    return ft * 3.048e-4

def deg_to_km(deg):
    conv = 2 * m.pi * 6.378e3 / 360
    return deg * conv

def bin_traffic(file, res_lon, res_lat, stat):
    lon_lat_grid = stats.binned_statistic_2d(np.array(file['Longitude']).flatten(),
                                             np.array(file['Latitude']).flatten(),
                                             np.array(file['Flown distance']).flatten(),
                                             statistic = stat,
                                             bins = [int((max_lon - min_lon) / res_lon), int((max_lat - min_lat) / res_lat)],
                                             range = [[min_lon, max_lon], [min_lat, max_lat]])
    
    return lon_lat_grid.statistic

dt = 120 # time window in minutes
start_date = '01-03-2015 12:00:00'
start_date = datetime.strptime(start_date, '%d-%m-%Y %H:%M:%S')
end_date = start_date + timedelta(minutes = dt)

filtered_flights = flights[flights['Time Over'].between(start_date, end_date)]

flight_IDs = filtered_flights['ID'].unique()

air_dist = []

for flight in flight_IDs:
    print(np.where(flight_IDs == flight))
    datalocs = filtered_flights[filtered_flights['ID'] == flight]
    delta_lon = list(map(dist, datalocs['Longitude'], datalocs['Longitude'].shift()))
    delta_lon = np.where(np.isnan(delta_lon), 0, delta_lon)
    delta_lat = list(map(dist, datalocs['Latitude'], datalocs['Latitude'].shift()))
    delta_lat = np.where(np.isnan(delta_lat), 0, delta_lat)
    delta_h = list(map(dist, datalocs['Flight Level'], datalocs['Flight Level'].shift()))
    delta_h = np.where(np.isnan(delta_h), 0, delta_h)
    d = np.sqrt(fl_to_km(delta_h)**2 + deg_to_km(delta_lon)**2 + deg_to_km(delta_lat)**2)
    d = d / (deg_to_km(res_lon) * deg_to_km(res_lat)) / dt * 60 * 24 # convert to km/(km^2 h)
    air_dist.append(d)

filtered_flights['Flown distance'] = np.concatenate(air_dist)

flights_WATD = bin_traffic(filtered_flights, res_lon, res_lat, stat = 'sum')

fig = plt.figure(figsize=(16,12))
ax = plt.subplot(111)
plt.title("Air Traffic Density (km/km$^2$ h) from {0} till {1}".format(start_date, end_date))
plt.rcParams.update({'font.size': 16})
bmap = Basemap(projection="lcc",resolution="i",width=5E6,height=3.5E6,
                             lon_0=15,lat_0=47.5,fix_aspect=False)
bmap.drawcoastlines()
bmap.drawcountries()
bmap.bluemarble()

#create and draw meridians and parallels grid lines
bmap.drawparallels(np.arange( -90., 120.,10.),labels=[1,0,0,0],fontsize=10)
bmap.drawmeridians(np.arange(-180.,180.,10.),labels=[0,0,0,1],fontsize=10)

lons, lats = bmap(*np.meshgrid(np.arange(min_lon + res_lon/2, max_lon, res_lon),
                              np.arange(min_lat + res_lat/2, max_lat, res_lat)))

# contourf 
im = bmap.contourf(lons, lats, flights_WATD.T, np.arange(0, 15, 0.1), 
                  extend='max', alpha = 0.7, cmap='jet')
cbar = plt.colorbar(im)


#%%
'''
--------------------------------GET ENGINE DATA--------------------------------
'''
AC_types = flights['AC_type'].unique()

engine = []

for AC_type in AC_types:
    # Constracting http query
    url = 'https://contentzone.eurocontrol.int/aircraftperformance/details.aspx?ICAO={0}&ICAOFilter={0}'.format(AC_type)
    # For avoid 403-error using User-Agent
    req = urllib.request.Request(url)
    response = urllib.request.urlopen(req)
    html = response.read()
    soup = BeautifulSoup(html, 'html.parser')
    # Extracting number of results
    engine.append(str(soup.find(id='MainContent_wsLabelPowerPlant').text))

AC_engine_tuple = list(zip(AC_types,engine)) # match AC type with engine type in tuple-like

AC_engine_data = pd.DataFrame(AC_engine_tuple, columns = ['AC_type', 'engine_type']) # convert to dataframe
AC_engine_data['engine_type'] = AC_engine_data['engine_type'].replace(['No data'], '')
AC_engine_data['engine_type'] = AC_engine_data['engine_type'].replace(['no data'], '')
engine_data_split = AC_engine_data['engine_type'].str.split(' or ', expand = True) # split engine_type col on possibilities

AC_engine_data = pd.concat([AC_engine_data['AC_type'], engine_data_split[0]], axis = 1) # combine AC type col with first engine type col
AC_engine_data.columns = ['AC_type', 'engine_type']

def get_engine_nr(row):
    try:
        lhs, rhs = row.split("x ", 1) # retrieve nr of engines
        lhs = lhs.split()[-1]
    except:
        lhs, rhs = '', ''
    return lhs, rhs

AC_engine_data['nr engines'] = AC_engine_data['engine_type'].apply(lambda x: get_engine_nr(x)[0])
AC_engine_data['engine_info'] = AC_engine_data['engine_type'].apply(lambda x: get_engine_nr(x)[1])

def remove_chars(row):
    try:
        row = re.sub('[ ,.!@#$-]', '', row)
    except:
        row = row
    return row

AC_engine_data['engine_info'] = AC_engine_data['engine_info'].apply(lambda x: remove_chars(x))

emissions = pd.read_csv(path + "Flight_data\\edb-emissions-databank.csv", encoding='latin-1')

emissions['Engine Identification'] = emissions['Engine Identification'].apply(lambda x: remove_chars(x))
#emissions['Engine Identification'] = emissions['Engine Identification'].replace(np.nan, '')

search_engine = pd.DataFrame()

for idx, item in emissions.iterrows():
    print(idx)
    search = AC_engine_data['engine_info'].str.contains(str(item['Engine Identification']),
                                                      flags=re.IGNORECASE, regex=True)
    search_engine[str(idx)] = search

search_cols = search_engine.apply(lambda x: search_engine.columns[x == True], axis = 1)

found_cols = pd.DataFrame([list(map(int, i.tolist())) for i in search_cols]) # convert list of indices of rows
                                                                # where engine identifier is found to df

fuel_flow = pd.Series(emissions['Fuel Flow Idle (kg/sec)'],index=emissions.index).to_dict()
fuel_flow = found_cols.replace(fuel_flow)
fuel_flow = fuel_flow.mean(axis=1, skipna = True)

AC_data = pd.concat([AC_engine_data['AC_type'], AC_engine_data['nr engines'], fuel_flow], axis = 1)
AC_data.columns = ['AC_type', 'nr engines', 'fuel flow']
AC_data['nr engines'] = pd.to_numeric(AC_data['nr engines'], downcast='integer')
AC_data['total ff'] = AC_data['nr engines'] * AC_data['fuel flow']