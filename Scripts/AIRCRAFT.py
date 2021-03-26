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
import os
os.environ['PROJ_LIB'] = r"C:\Users\fafri\anaconda3\pkgs\proj4-6.1.1-hc2d0af5_1\Library\share"
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.animation as animation
from scipy import stats
import math as m
import urllib
from bs4 import BeautifulSoup
import re
from datetime import date, timedelta, datetime
#from SPACE_processing import hdf4_files, CALIPSO_analysis

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

path = "E:\\Research_cirrus\\"

res_lon = 0.25 # set desired resolution in longitudinal direction in degs
res_lat = 0.25 # set desired resolution in latitudinal direction in degs
min_lon = -10 # research area min longitude
max_lon = 40 # research area max longitude
min_lat = 35 # research area min latitude
max_lat = 60 # research area max latitude

#%%

class flight_analysis: # e.g. testflight = flight_analysis('Flights_20150301_20150331.csv', 30, 'load', 'interpolated_flightdata_0315_part_I.pkl')
                        # or testflight = flight_analysis('Flights_20150301_20150331.csv', 30, 'perform', 'firsttest.pkl', dates = test.dates, times = test.times, lon_pos = test.lon_pos, lat_pos = test.lat_pos)
    def __init__(self, filename, window, option, savename, **calipso_input): # option can be 'perform' or 'load' (interpolated data), savename its filename
        self.filename = filename
        self.option = option
        self.savename = savename
        self.window = window
        if calipso_input:
            self.dates = calipso_input['dates']
            self.times = calipso_input['times']
            self.lon_pos = calipso_input['lon_pos']
            self.lat_pos = calipso_input['lat_pos']
        # automatically run essential methods
        self.individual_flights()
        self.flighttracks()
        self.interpol_exe()
        self.merge_datasets()
        self.grid_ATD()
            
    def dist(self, arr1, arr2):
        return abs(arr1 - arr2)
    
    def fl_to_km(self, fl):
        ft = fl * 100
        return ft * 3.048e-4
    
    def deg_to_km(self, deg):
        conv = 2 * m.pi * 6.378e3 / 360
        return deg * conv
    
    def individual_flights(self):
        '''
        -------------------------------import and format------------------------------
        '''
        # import flights
        try:
            self.flights = pd.read_csv(path + "Flight_data\\" + self.filename, sep=";")
            # rename columns
            self.flights.columns = ["ID", "ICAO_dep", "lat_dep", "lon_dep", "ICAO_dest", "lat_dest", "lon_dest",
                                "planned_dep_time", "planned_arr_time", "dep_time", "arr_time", "AC_type",
                                "AC_operator", "AC_regis", "flight_type", "market", "req_FL", "dist"]
        except:
            self.flights = pd.read_csv(path + "Flight_data\\" + self.filename, sep=",")
            # rename columns
            self.flights.columns = ["ID", "ICAO_dep", "lat_dep", "lon_dep", "ICAO_dest", "lat_dest", "lon_dest",
                                "planned_dep_time", "planned_arr_time", "dep_time", "arr_time", "AC_type",
                                "AC_operator", "AC_regis", "flight_type", "market", "req_FL", "dist"]
        
        print("Original dataset consists of {0} flights.".format(len(self.flights)))
        
        '''
        -----------------------------------filtering-----------------------------------
        '''
        
        # invalid values
        crucial_cols = ["ID", "ICAO_dep", "lat_dep", "lon_dep", "ICAO_dest", "lat_dest", "lon_dest",
                                "dep_time", "arr_time", "AC_type", "AC_operator", "flight_type"] # columns that should not contain NaN
        nan_flights = self.flights[self.flights[crucial_cols].isna().any(axis=1)]
        
        print("Number of flights with important missing data is {0}.".format(len(nan_flights)))
        
        unknown_dep_airports = nan_flights[nan_flights["lat_dep"].isna()]["ICAO_dep"].unique()
        unknown_dest_airports = nan_flights[nan_flights["lat_dest"].isna()]["ICAO_dest"].unique()
        unknown_airports = list(set(unknown_dep_airports) | set(unknown_dest_airports))
        
        # recover locations of known airports (ZZZZ and AFIL are not locatable!)
        self.flights.loc[self.flights["ICAO_dep"] == "FAOR", "lat_dep"] = -26
        self.flights.loc[self.flights["ICAO_dep"] == "FAOR", "lon_dep"] = 28
        self.flights.loc[self.flights["ICAO_dest"] == "FAOR", "lat_dest"] = -26
        self.flights.loc[self.flights["ICAO_dest"] == "FAOR", "lon_dest"] = 28
        
        self.flights.loc[self.flights["ICAO_dep"] == "GQNN", "lat_dep"] = 18
        self.flights.loc[self.flights["ICAO_dep"] == "GQNN", "lon_dep"] = -15
        self.flights.loc[self.flights["ICAO_dest"] == "GQNN", "lat_dest"] = 18
        self.flights.loc[self.flights["ICAO_dest"] == "GQNN", "lon_dest"] = -15
        
        nan_flights = self.flights[self.flights[crucial_cols].isna().any(axis=1)]
        
        print("Number of flights with important missing data after recovering known airport locations is {0}.".format(len(nan_flights)))
        
        self.flights = self.flights[~self.flights[crucial_cols].isna().any(axis=1)]
        
        print("Number of flights after filtering those rows is {0}.".format(len(self.flights)))
        
        # not so interesting columns that can be dropped
        droplabels = ["lat_dep", "lon_dep", "lat_dest", "lon_dest", "planned_dep_time", "planned_arr_time",
                      "dep_time", "arr_time", "req_FL"]
        
        self.flights = self.flights.drop(labels = droplabels, axis = 1)
        
        return self.flights
    
    def airports(self):
        airports = pd.read_csv(path + "Additional_data\\airports.csv", sep=",", encoding='latin-1')
        
        airports = airports[["airport", "ICAO"]]
        
        # change colname to merge on departure airport
        airports.columns = ["airport", "ICAO_dep"]
        self.flights = self.flights.merge(airports, how = "inner", on = ["ICAO_dep"])
        
        # change colname to merge on arrival airport
        airports.columns = ["airport", "ICAO_dest"]
        self.flights = self.flights.merge(airports, how = "inner", on = ["ICAO_dest"])
        
        self.flights.columns = ["ID", "ICAO_dep", "ICAO_dest", "AC_type", "AC_operator",
                                "AC_regis", "flight_type", "market", "dist", "dep_airport", "dest_airport"]
        
        airport_count = self.flights.groupby(by = "dep_airport")
        airport_count = airport_count.size().rename("count")
        airport_count = airport_count.sort_values(ascending = False)
        
        fig, ax = plt.subplots()
        ax.tick_params(axis='y', labelsize = 10, rotation = 30)
        ax.tick_params(axis='x', labelsize = 14, rotation = 0)
        ax.set_title("Number of flights departing in March 2015")
        ax.barh(airport_count.index[0:20], airport_count[0:20])
        ax.set_xticks(np.arange(0, 22000, 2000))
        ax.set_facecolor('darkgrey')
    
        return self.flights
    
    '''
    -------------------------------RADAR FLIGHTS-----------------------------------
    '''
    
    def flighttracks(self):
        
        self.flighttrack = pd.read_csv(path + "Flight_data\\Flight_Points_Actual_" + self.filename[-21:], sep=",")

        print("Original dataset consists of {0} flightpoints.".format(len(self.flighttrack)))
        
        # check for NaN values
        nan_flightpoint = self.flighttrack[self.flighttrack.isna().any(axis=1)]
        
        # all NaN values are in cols Latitude & Longitude -> drop
        self.flighttrack = self.flighttrack.dropna(how = 'any')
        
        print("Number of flight points after dropping NaN locations is {0}.".format(len(self.flighttrack)))
        
        # only keep points within ROI
        self.flighttrack = self.flighttrack[(self.flighttrack["Latitude"] > min_lat) & 
                                            (self.flighttrack["Latitude"] < max_lat) & 
                                            (self.flighttrack["Longitude"] > min_lon) & 
                                            (self.flighttrack["Longitude"] < max_lon)]
        
        print("Number of flight points after selecting ROI is {0}.".format(len(self.flighttrack)))
        
        # only keep data above 6 km (where cirrus can form)
        self.flighttrack = self.flighttrack[self.flighttrack["Flight Level"] > 197]
        
        print("Number of flight points after omitting points below 6 km is {0}.".format(len(self.flighttrack)))
        
        self.flighttrack['Time Over'] = pd.to_datetime(self.flighttrack['Time Over'], format="%d-%m-%Y %H:%M:%S")
    
        return self.flighttrack
    
    def map_movements(self, start_date, end_date): # in string format, eg '05-03-2015 12:00:00'
        '''
        -----------------------map movements between two dates-------------------------
        '''
        
        start_date = datetime.strptime(start_date, '%d-%m-%Y %H:%M:%S')
        end_date = datetime.strptime(end_date, '%d-%m-%Y %H:%M:%S')
        
        filtered_flights = self.flighttrack[self.flighttrack['Time Over'].between(start_date, end_date)]
        
        flight_IDs = self.flighttrack["ECTRL ID"].unique()
        
        fig = plt.figure(figsize=(16,12))
        ax = plt.subplot(111)
        plt.title("All registered flights (after filtering) between {0} and {1}".format(start_date, end_date))
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
            
    def dist2sat(self, df, current_lon, current_lat):

        d_lon = self.deg_to_km(abs(df['Longitude'] - current_lon))
        d_lat = self.deg_to_km(abs(df['Latitude'] - current_lat))
        
        return np.sqrt(d_lon**2 + d_lat**2)
    
    def interpol(self, ID):
        '''
        --------------------INTERPOLATION TO 5 MIN INTERVALS--------------------------
        '''
        
        data = self.selected_flightpoints[self.selected_flightpoints['ECTRL ID'] == ID][['Time Over', 'Flight Level', 'Longitude', 'Latitude', 'ID']]
        interp = data.set_index('Time Over').resample('5min').mean().interpolate(method='linear')
        interp = interp.reset_index()
        interp['ID'] = [ID] * len(interp)
        
        return interp
    
    def interpol_exe(self): #dates, times, window, lon_pos, lat_pos
        self.combined_datetime = [datetime.combine(date, time) for date, time 
                                 in zip(self.dates, self.times)]
        if self.option == 'perform':

            radius = self.window / 60 * 1200 + 100 # filter aircraft within certain radius defined by nr of hours times max expected flight speed in km/h (incl a margin)
            long_pos = [np.mean(sample) for sample in self.lon_pos]
            lat_pos = [np.mean(sample) for sample in self.lat_pos]
            
            cols = np.concatenate([self.flighttrack.columns, ['Dist-2-Sat']])
            
            self.selected_flightpoints = pd.DataFrame(columns = cols)
                        
            for idx, current_date in enumerate(self.combined_datetime):
                time_open = current_date - timedelta(minutes = self.window)
                filtered_flights = self.flighttrack[self.flighttrack['Time Over'].between(time_open, current_date)]
                current_lon = long_pos[idx]
                current_lat = lat_pos[idx]
                filtered_flights['Dist-2-Sat'] = filtered_flights[['Latitude', 'Longitude']].apply(lambda x: self.dist2sat(x, current_lon, current_lat), axis = 1)
                self.selected_flightpoints = self.selected_flightpoints.append(filtered_flights[filtered_flights['Dist-2-Sat'] < radius],
                                                         ignore_index = True)
    
            self.selected_flightpoints.drop_duplicates() # remove possible duplicates

            cols = ['Time Over', 'Flight Level', 'Longitude', 'Latitude',  'ID']
            self.flight_interpol = pd.DataFrame(columns=cols)
            
            print(len(self.selected_flightpoints['ECTRL ID'].unique()))
            
            for idx, flight in enumerate(self.selected_flightpoints['ECTRL ID'].unique()):
                    
                self.flight_interpol = self.flight_interpol.append(self.interpol(flight), 
                                                                   ignore_index = True)
                print(idx)
                
            self.flight_interpol.to_pickle(self.savename)
        elif self.option == 'load':
            try:
                self.flight_interpol = pd.read_pickle(path + 'Flight_data\\' + self.savename) #to load 123.pkl back to the dataframe df
            except:
                print("File not found.")
        else:
            print("Option not recognized.")
            
        return self.flight_interpol
    
    def merge_datasets(self):
        '''
        -------------------------------MERGE DATASETS----------------------------------
        '''
        
        self.flights = self.flight_interpol.merge(self.flights, how = "inner", on = ["ID"])
        
        return self.flights
    
    def perform_animation(self, start_date, end_date, save): # start and end date in string format, save as boolean
        '''
        -------------------------------ANIMATE FLIGHTS--------------------------------
        '''
        
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
            
        x, y = map(0, 0)
        point = map.plot(x, y, 'ro', markersize = 3, color = 'red')[0]
        
        text = plt.text(0.5, 1.05, str(start_date), ha="center", transform=ax.transAxes,)
        
        flights = self.flights
        def animate(i):

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

        if save == True:
            # Set up formatting for the movie files
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
            myAnimation.save('aircraft.mp4', writer=writer)
    
    '''
    ---------------------------GRID AIR TRAFFIC (ATD)------------------------------
    '''
    
    def bin_traffic(self, file, res_lon, res_lat, stat):
        lon_lat_grid = stats.binned_statistic_2d(np.array(file['Longitude']).flatten(),
                                                 np.array(file['Latitude']).flatten(),
                                                 np.array(file['Flown distance']).flatten(),
                                                 statistic = stat,
                                                 bins = [int((max_lon - min_lon) / res_lon), int((max_lat - min_lat) / res_lat)],
                                                 range = [[min_lon, max_lon], [min_lat, max_lat]])
        
        return lon_lat_grid.statistic
    
    def flown_distance(self, start_date, dt): # start date as datetime obj, dt in minutes
    
        end_date = start_date + timedelta(minutes = dt)
        
        self.filtered_flights = self.flights[self.flights['Time Over'].between(start_date, end_date)]
        
        flight_IDs = self.filtered_flights['ID'].unique()

        self.air_dist = []
        
        for flight in flight_IDs:
            datalocs = self.filtered_flights[self.filtered_flights['ID'] == flight]
            delta_lon = list(map(self.dist, datalocs['Longitude'], datalocs['Longitude'].shift()))
            delta_lon = np.where(np.isnan(delta_lon), 0, delta_lon)
            delta_lat = list(map(self.dist, datalocs['Latitude'], datalocs['Latitude'].shift()))
            delta_lat = np.where(np.isnan(delta_lat), 0, delta_lat)
            delta_h = list(map(self.dist, datalocs['Flight Level'], datalocs['Flight Level'].shift()))
            delta_h = np.where(np.isnan(delta_h), 0, delta_h)
            d = np.sqrt(self.fl_to_km(delta_h)**2 + self.deg_to_km(delta_lon)**2 + self.deg_to_km(delta_lat)**2)
            d = d / (self.deg_to_km(res_lon) * self.deg_to_km(res_lat)) / dt * 60 * 24 # convert to km/(km^2 h)
            self.air_dist.append(d)
        
        try:
            self.air_dist = np.concatenate(self.air_dist)
        except:
            self.air_dist = []
        
        self.filtered_flights['Flown distance'] = self.air_dist
        
        return self.filtered_flights
    
    def time_gaps(self):
        
        self.timegaps = []
        for idx, flight in enumerate(self.flighttrack['ECTRL ID'].unique()):
            self.time_over = self.flighttrack[self.flighttrack['ECTRL ID'] == flight][['Time Over']]
            timedist = list(map(self.dist, self.time_over['Time Over'][1:], self.time_over['Time Over'].shift()[1:]))
            self.timegaps.append([timegap.seconds // 60 for timegap in timedist])
            print(idx)
            if idx == 5000:
                self.timegaps = np.concatenate(self.timegaps)
                fig, ax1 = plt.subplots(figsize=(8, 4))
                ax1.hist(self.timegaps, bins = 60, color = "purple", alpha = 0.3)
                ax1.set_xlabel("time gap in mins between two registered flight points")
                ax1.set_ylabel("occurrence count")
                ax1.grid(False)
                ax2 = ax1.twinx()
                ax2.hist(self.timegaps, bins = 60, density=True,
                           cumulative=True, color = "#f39c12", alpha = 0.3)
                ax2.set_ylabel("likelihood of $t_{gap} <= val$")
                ax2.grid(False)
                plt.show()
                break
            
    def grid_ATD(self):
        
        self.flights_ATD = []
        
        for date_window in self.combined_datetime:
            start_date = date_window - timedelta(minutes = self.window)
            self.filtered_flights = self.flown_distance(start_date, self.window)          
            self.flights_ATD.append(self.bin_traffic(self.filtered_flights, res_lon, res_lat, stat = 'sum'))
            
        self.flights_ATD = np.stack(self.flights_ATD)
    
        return self.flights_ATD
    
    def plot_ATD(self, idx):
        
        fig = plt.figure(figsize=(16,12))
        ax = plt.subplot(111)
        plt.title("Air Traffic Density (km/km$^2$ h) from {0} till {1}".format(self.combined_datetime[idx] - timedelta(minutes = self.window), 
                                                                               self.combined_datetime[idx]))
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
        im = bmap.contourf(lons, lats, self.flights_ATD[idx].T, np.arange(0, 15, 0.1), 
                          extend='max', cmap='jet')
        cbar = plt.colorbar(im)
    
    '''
    -------------------------WATD (not yet approved)-------------------------------
    '''
    
    def get_engine_nr(self, row):
        try:
            lhs, rhs = row.split("x ", 1) # retrieve nr of engines
            lhs = lhs.split()[-1]
        except:
            lhs, rhs = '', ''
        return lhs, rhs
    
    def remove_chars(self, row):
        try:
            row = re.sub('[ ,.!@#$-/:&]', '', row)
        except:
            row = row
        return row
    
    def AC_fuel_flow(self):
        
        AC_types = self.flights['AC_type'].unique()
        
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
        
        AC_engine_data['nr engines'] = AC_engine_data['engine_type'].apply(lambda x: self.get_engine_nr(x)[0])
        AC_engine_data['engine_info'] = AC_engine_data['engine_type'].apply(lambda x: self.get_engine_nr(x)[1])
        
        AC_engine_data['engine_info'] = AC_engine_data['engine_info'].apply(lambda x: self.remove_chars(x))
        
        emissions = pd.read_csv(path + "Flight_data\\edb-emissions-databank.csv", encoding='latin-1')
        
        emissions['Engine Identification'] = emissions['Engine Identification'].apply(lambda x: self.remove_chars(x))
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
        
        self.AC_data = pd.concat([AC_engine_data['AC_type'], AC_engine_data['nr engines'], fuel_flow], axis = 1)
        self.AC_data.columns = ['AC_type', 'nr engines', 'fuel flow']
        self.AC_data['nr engines'] = pd.to_numeric(self.AC_data['nr engines'], downcast='integer')
        self.AC_data['total ff'] = self.AC_data['nr engines'] * self.AC_data['fuel flow']
        
        return self.AC_data