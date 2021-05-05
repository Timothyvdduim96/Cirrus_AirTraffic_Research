# -*- coding: utf-8 -*-
"""
@author: fafri
"""
'''
------------------------------PACKAGES-----------------------------------------
'''

from netCDF4 import Dataset
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as m
from scipy import stats
import re
import random
import urllib
import fnmatch
import lxml.html
import time 
import cdsapi

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
--------------------------------PARAMS-----------------------------------------
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
---------------------------------PATHS-----------------------------------------
'''

flight_path = 'E:\\Research_cirrus\\Flight_data\\'
lidar_path = 'E:\\Research_cirrus\\CALIPSO_data\\'
meteo_path = 'E:\\Research_cirrus\\ERA5_data\\'
meteosat_path = 'E:\\Research_cirrus\\Meteosat_CLAAS_data\\'

#%%
'''
---------------EXTRACT ERA5 REANALYSIS PRESSURE LEVEL DATA---------------------
'''

class ERA5_reanalysis:
    
    '''
        Parameters
        ----------
        month : month, str
        year : year, str
        pres_lvls : pressure levels, list of str ended with a comma
        savename : name new file, list of str
        
        Returns
        -------
        API requested meteo data, RH and temp
    '''
    
    def __init__(self, month, year, pres_lvls, savename):
        self.month = month
        self.year = year
        self.pres_lvls = pres_lvls
        self.savename = savename
        self.api_req()
        
    def api_req(self):
        c = cdsapi.Client()
    
        c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'relative_humidity', 'temperature',
            ],
            'pressure_level': self.pres_lvls,
            'year': self.year,
            'month': self.month,
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': [
                max_lat, min_lon, min_lat,
                max_lon,
            ],
        },
        '{0}.nc'.format(self.savename))

#%%
'''
---------------------------HANDY FUNCTIONS-------------------------------------
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
        
    def piecewise_linear(x, x0, y0, k1, k2):
        return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
    
    def remove_chars(row):
        try:
            row = re.sub('[ ,.!@#$-/:&]', '', row)
        except:
            row = row
        return row
    
    def alt(h_b, P_ref, T_ref, P):
                
        M = 0.0289644 # molar mass Earths air in kg/mol
        R = 8.3144598 # universal gas constant in J/(mol K)
        g_0 = 9.80665 # gravitational constant
        
        return (R * T_ref / (g_0 * M) * np.log(P_ref / P) + h_b) / 1e3
    
    def coeff_linregr(data, x, Neff):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,data)
        t = r_value*np.sqrt(Neff - 2)/np.sqrt(1 - r_value**2) #compute t for t test
        p_val = stats.t.sf(np.abs(t), Neff-1) * 2 #compute p value from t and dimensions
        return slope, std_err, p_val
    
    def hour_rounder(t):
            # Rounds to nearest hour by adding a timedelta hour if minute >= 30
            return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
                       + datetime.timedelta(hours=t.minute//30))
        
    def quarterhr_rounder(t):
            # Rounds to nearest hour by adding a timedelta hour if minute >= 30
            return t.replace(second=0, microsecond=0, minute=0,
                              hour=t.hour) + datetime.timedelta(minutes=round(t.minute / 15) * 15)
    
    def convert_seconds(n): 
        timestr = time.strftime('%H:%M:%S', time.gmtime(n))
        return datetime.datetime.strptime(timestr, '%H:%M:%S').time()
    
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

    
    def url_list(url):
        urls = []
        connection = urllib.request.urlopen(url)
        dom =  lxml.html.fromstring(connection.read())
        for link in dom.xpath('//a/@href'):
            urls.append(link)
        return urls
    
    def gen_dirlist(self, dirlist, savename):
        # wget --load-cookies {path}.urs_cookies --save-cookies {path}.urs_cookies --auth-no-challenge=on --keep-session-cookies --user={username} --ask-password --header "Authorization: Bearer {token}" --content-disposition -i {path}\{savename}.dat -P {path}
        for idx in range(len(dirlist)):
            
            urls = self.url_list(dirlist[idx])
            
            filetype = "*.hdf"
            file_list = [filename for filename in fnmatch.filter(urls, filetype)]
                
            with open('{0}.dat'.format(savename), 'a') as text_file:
                for file in file_list:
                    name = '{0}{1}'.format(dirlist[idx], file)
                    text_file.write(name + '\n')
    
    def multidim_slice(ind_arr, arr):
        output_arr = np.zeros((np.shape(arr)[1], np.shape(arr)[2]))
        for row in range(len(ind_arr)):
            for col in range(len(ind_arr[row])):
                ind = ind_arr[row, col]
                if np.isnan(ind) == True:
                    output_arr[row, col] = np.mean(arr[:, row, col])
                else:
                    ind = int(ind - 1)
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
    
    def bin_1D(x_list, var, stat):
        bins_pressureL = stats.binned_statistic(np.array(x_list).flatten(),
                                     np.array(var).flatten(),
                                     statistic = stat,
                                     bins = int(len(x_list) - 1),
                                     range = [min(x_list), max(x_list)]) # pressure levels in upper troposphere include 100-400 in steps of 50hPa (see SPACE_import)
    
        return bins_pressureL.statistic
    
    def bin_2d(x_list, y_list, z_list, min_x, max_x, min_y, max_y, res_x, res_y, stat):
        grid = stats.binned_statistic_2d(np.array(x_list).flatten(),
                                                 np.array(y_list).flatten(),
                                                 np.array(z_list).flatten(),
                                                 statistic = stat,
                                                 bins = [int((max_x - min_x) / res_x), int((max_y - min_y) / res_y)],
                                                 range = [[min_x, max_x], [min_y, max_y]])
        
        return grid.statistic
        
    