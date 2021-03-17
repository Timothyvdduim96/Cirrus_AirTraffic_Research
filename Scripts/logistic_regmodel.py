# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:09:13 2021

@author: T Y van der Duim
"""

'''
------------------------------PACKAGES-----------------------------------------
'''

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np

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

def netcdf_import(file):
    data = Dataset(file,'r')
    print(data.variables)
    print(data.variables.keys())
    my_dict = {}
    for key in data.variables.keys():
        if key == 'longitude' or key == 'latitude' or key == 'time' or key == 'level':
            my_dict[key] = data[key][:]
        else:
            my_dict[key] = data[key][:, :, :, :]
    return my_dict

#%%

'''
-----------------------------LOGISTIC REGMODEL---------------------------------
'''

path = 'E:\\Research_cirrus\\'

file = path + "\\ERA5_data\\ERA5_jan15_train.nc"

pres_lev = []
train_data = netcdf_import(file)

rel_hum_train = np.mean(train_data['r'], axis = 1)
temp_train = np.mean(train_data['t'], axis = 1)

rel_hum_train = rel_hum_train.reshape(-1)
temp_train = temp_train.reshape(-1)

corr = np.corrcoef(rel_hum_train, temp_train) # correlation between rel. hum. and temp

X_train = np.column_stack((rel_hum_train, temp_train))

logreg = LogisticRegression()
logreg.fit(X_train, y_train)