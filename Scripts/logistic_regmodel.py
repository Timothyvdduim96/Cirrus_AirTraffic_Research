# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:09:13 2021

@author: T Y van der Duim
"""

'''
------------------------------PACKAGES-----------------------------------------
'''

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
from SPACE_processing import *

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
test = cirrus_ani[0]
path = 'E:\\Research_cirrus\\'

def model_arrays(file, date_start, date_end, nr_periods):
    pres_lev = []
    train_data = netcdf_import(file)
    
    rel_hum_train = train_data['r']
    print(rel_hum_train.shape)
    temp_train = train_data['t']
    
    # select parts of ERA5 data at times when CALIPSO overpasses Europe
    overpass_time = [datetime.combine(date, time) 
                     for date, time in zip(dates, times)] # merge dates and times to get a datetime list of Calipso overpasses
    
    start_train = datetime.strptime(date_start, '%Y-%m-%d %H:%M')
    end_train = datetime.strptime(date_end, '%Y-%m-%d %H:%M')
    
    hour_list = pd.date_range(start = start_train, end = end_train, periods = nr_periods).to_pydatetime().tolist()
    
    idx = [key for key, val in enumerate(hour_list) if val in overpass_time]
    print(idx)
    rel_hum_train = rel_hum_train[idx, :, :, :]
    print(rel_hum_train.shape)
    temp_train = temp_train[idx, :, :, :]
    
    def duplicates(lst, item):
       return [i for i, x in enumerate(lst) if x == item]
    
    remove_dup = dict((x, duplicates(overpass_time, x))[1] for x in set(overpass_time) 
                      if overpass_time.count(x) > 1)
    
    cirrus_cover = np.delete(cirrus_ani, list(remove_dup.values()), axis = 0)
    # transpose 2nd and 3rd axes (lat and lon) so that it matches the cirrus cover array and match size
    rel_hum_train = np.transpose(rel_hum_train, (0, 1, 3, 2))[:, :, :-1, :-1] 
    temp_train = np.transpose(temp_train, (0, 1, 3, 2))[:, :, :-1, :-1]
    rel_hum_train = np.where(np.isnan(cirrus_cover) == True, np.nan, rel_hum_train)
    temp_train = np.where(np.isnan(cirrus_cover) == True, np.nan, temp_train)
    
    # flatten the arrays (1D)
    cirrus_cover = cirrus_cover.reshape(-1)
    rel_hum_train = rel_hum_train.reshape(-1)
    temp_train = temp_train.reshape(-1)
    
    # remove nan instances
    cirrus_cover = cirrus_cover[~(np.isnan(cirrus_cover))] 
    rel_hum_train = rel_hum_train[~(np.isnan(rel_hum_train))] 
    temp_train = temp_train[~(np.isnan(temp_train))] 
    
    corr = np.corrcoef(rel_hum_train, temp_train) # correlation between rel. hum. and temp
    
    X_train = np.column_stack((rel_hum_train, temp_train))
    
    y_train = np.where(cirrus_cover > 0, 1, 0) # convert cirrus cover into binary response (cirrus or no cirrus) with certain thr
    return X_train, y_train

X_train, y_train = model_arrays(path + "ERA5_data\\ERA5_jan15_train.nc", "2015-01-01 00:00",
                              "2015-01-31 23:00", 744)

# logreg model (https://realpython.com/logistic-regression-python/)
model = LogisticRegression(class_weight = 'balanced').fit(X_train, y_train)

# evaluate model
model.classes_
model.predict_proba(X_train) # 1st col is probability of predicted output being zero, second = 1 - first
model.predict(X_train) # class predictions
model.score(X_train, y_train)

# testdata
X_test, y_test = model_arrays(path + "ERA5_data\\ERA5_feb15_test.nc", "2015-02-01 00:00",
                              "2015-02-28 23:00", 672)

model.score(X_test, y_test)
#%%
cm = confusion_matrix(y_test, model.predict(X_test))

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()