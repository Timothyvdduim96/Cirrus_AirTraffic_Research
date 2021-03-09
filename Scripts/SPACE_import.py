# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:46:25 2021

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
from requests import Session
import sys
import urllib
import fnmatch
import lxml.html
import wget
import keyring
from scipy import stats
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

#%%
'''
---------------EXTRACT ERA5 REANALYSIS PRESSURE LEVEL DATA---------------------
'''

def api_req():
    c = cdsapi.Client()

    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'relative_humidity', 'temperature',
            ],
            'pressure_level': [
                '100', '150', '200',
                '250', '300', '350',
                '400',
            ],
            'year': '2017',
            'month': [
                '03', '06', '09',
                '12',
            ],
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
                60, -10, 35,
                40,
            ],
        },
        'ERA5_17.nc')

api_req()

#%%
'''
---------------------------EXTRACT CALIPSO DATA--------------------------------
'''

dirlist = ['https://xfr139.larc.nasa.gov/322beb85-0949-4679-ae8b-2a7d310a634a/',
           'https://xfr139.larc.nasa.gov/bc9a75f7-09fc-4d6a-a9d0-67071c8e23b2/',
           'https://xfr139.larc.nasa.gov/994fb799-eac2-4f83-9896-b9064885ce33/',
           'https://xfr139.larc.nasa.gov/6cbabe4b-26ed-451a-9dc2-27222301018f/',
           'https://xfr139.larc.nasa.gov/7a7fbb75-ad03-4f27-926b-1850b1726784/',
           'https://xfr139.larc.nasa.gov/9ad82758-fd27-4f77-bba6-711b9ddf2204/',
           'https://xfr139.larc.nasa.gov/f6d89b2c-4545-44be-8574-750ca8230098/',
           'https://xfr139.larc.nasa.gov/f0970b13-28a9-4078-961d-152742f2db2d/',
           'https://xfr139.larc.nasa.gov/b6f8ffd2-71e6-4e35-a0f8-052dd3d32bec/',
           'https://xfr139.larc.nasa.gov/8cb264c9-e4c4-4bde-8791-c0e21b64032e/',
           'https://xfr139.larc.nasa.gov/552f8d5a-36bb-4b3b-b0c8-039be64077b5/',
           'https://xfr139.larc.nasa.gov/64ee3b12-98c4-4a11-bd49-6803bdddffc7/',
           'https://xfr139.larc.nasa.gov/abc2079b-75f6-41ed-beb4-26e7439e9b4c/',
           'https://xfr139.larc.nasa.gov/b5642a37-c7a2-4844-9567-10e1f7162523/',
           'https://xfr139.larc.nasa.gov/81151960-0122-488d-8c59-5dfa82f1aafd/',
           'https://xfr139.larc.nasa.gov/e18211c1-93fd-4c29-afa8-08dfcf14d48e/',
           'https://xfr139.larc.nasa.gov/9fdc71b9-d311-424e-9107-9f341287e028/',
           'https://xfr139.larc.nasa.gov/fb6c769d-060a-4e80-ab65-9d05b05092ea/',
           'https://xfr139.larc.nasa.gov/f65b8006-bb26-43fb-9227-79bffee7ab56/',
           'https://xfr139.larc.nasa.gov/9d867f62-c01d-4029-bd4e-ad8a710f3d18/',
           'https://xfr139.larc.nasa.gov/4cbc7c75-310f-43b7-8f71-28690f31d050/',
           'https://xfr139.larc.nasa.gov/9c05d26a-f939-47a7-b4c6-e12dc7e76889/',
           'https://xfr139.larc.nasa.gov/e18bf041-018a-4fa9-b828-97769a8e00f5/',
           'https://xfr139.larc.nasa.gov/158a34b3-4269-4099-b5be-f5363b25f9ae/'] 
# 03/15 (DONE), 06/15 (DONE), 09/15 (DONE), 12/15, 03/16, 06/16, 09/16, 12/16, 03/17, 06/17,
# 09/17, 12/17, 03/18, 06/18, 09/18, 12/18, 03/19, 06/19, 09/19, 12/19, 03/20,
# 06/20, 09/20, 12/20

def url_list(url):
    urls = []
    connection = urllib.request.urlopen(url)
    dom =  lxml.html.fromstring(connection.read())
    for link in dom.xpath('//a/@href'):
        urls.append(link)
    return urls

directory = dirlist[3]

urls = url_list(directory)

filetype = "*.hdf"
file_list = [filename for filename in fnmatch.filter(urls, filetype)]

for file in file_list:
    url = '{0}{1}'.format(directory, file)
    print(file)
    filename = wget.download(url)





