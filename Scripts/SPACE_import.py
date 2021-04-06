# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:46:25 2021

@author: T Y van der Duim
"""

'''
------------------------------PACKAGES-----------------------------------------
'''

import matplotlib.pyplot as plt
import os
os.environ['PROJ_LIB'] = r"C:\Users\fafri\anaconda3\pkgs\proj4-6.1.1-hc2d0af5_1\Library\share"
import urllib
import fnmatch
import lxml.html
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
            '100', '125', '150',
            '175', '200', '225',
            '250', '300', '350',
            '400', '450',
        ],
        'year': '2015',
        'month': '01',
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
    'ERA5_01_15.nc')

api_req()

#%%
'''
---------------------------EXTRACT CALIPSO DATA--------------------------------
'''

dirlist = ["https://xfr139.larc.nasa.gov/f0c1d3dd-9071-4d87-82c4-73e845fbb6bb/",
           "https://xfr139.larc.nasa.gov/7adf3193-3520-41d3-b87e-591212dacdc9/",
           "https://xfr139.larc.nasa.gov/0773bd02-b303-44d5-bd7c-78aa64d25c65/",
           "https://xfr139.larc.nasa.gov/feda055f-9da5-4fe6-8e09-de1fc780bbcc/",
           "https://xfr139.larc.nasa.gov/4962dbbc-dcd4-457d-9174-3dac0e5b1a00/",
           "https://xfr139.larc.nasa.gov/5ee8e267-9b2b-4e08-a07b-cad818ada998/",
           "https://xfr139.larc.nasa.gov/ef570a9e-a0fb-4912-997b-c7bd920ee298/",
           "https://xfr139.larc.nasa.gov/f08b2f52-6852-467a-a08a-743359320202/"]
# 03/19, 06/19, 09/19, 12/19, 03/20, 06/20, 09/20, 12/20

def url_list(url):
    urls = []
    connection = urllib.request.urlopen(url)
    dom =  lxml.html.fromstring(connection.read())
    for link in dom.xpath('//a/@href'):
        urls.append(link)
    return urls

for idx in range(len(dirlist)):
    print(idx)
    urls = url_list(dirlist[idx])
    
    filetype = "*.hdf"
    file_list = [filename for filename in fnmatch.filter(urls, filetype)]
    #files.append(file_list)
        
    with open('file_03_19_12_20.dat', 'a') as text_file:
        for file in file_list:
            name = '{0}{1}'.format(dirlist[idx], file)
            text_file.write(name + '\n')

#%%
'''
--------------------------------RUN IN CMD-------------------------------------
'''

# wget --load-cookies C:\Users\fafri\.urs_cookies --save-cookies C:\Users\fafri\.urs_cookies --auth-no-challenge=on --keep-session-cookies --user=timomaster --ask-password --header "Authorization: Bearer c3bc08e7ead1b953f28ee12773dd6a9519f94457c801a1d4799f63af9efd2e90" --content-disposition -i C:\Users\fafri\file_03_19_12_20.dat -P C:\Users\fafri\





