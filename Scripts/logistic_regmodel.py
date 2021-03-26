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
import datetime

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
-----------------------------LOGISTIC REGMODEL---------------------------------
'''

path = 'E:\\Research_cirrus\\'

def model_arrays(file, calipso_folder, date_start, date_end, nr_periods):

    data = netcdf_import(file)
    
    rel_hum = data['r']
    print(rel_hum.shape)
    temp = data['t']
    
    calipso = CALIPSO(path + "CALIPSO_data\\" + calipso_folder, pressure_axis = True)
    # calipso_train = np.load('calipso_train.npy')
    # dates = np.load('dates_train.npy')
    # times = np.load('times_train.npy')
    
    # select parts of ERA5 data at times when CALIPSO overpasses Europe
    overpass_time = [datetime.combine(date, time) 
                      for date, time in zip(dates, times)] # merge dates and times to get a datetime list of Calipso overpasses
    
    start = datetime.strptime(date_start, '%Y-%m-%d %H:%M')
    end = datetime.strptime(date_end, '%Y-%m-%d %H:%M')
    
    hour_list = pd.date_range(start = start, end = end, periods = nr_periods).to_pydatetime().tolist()
    
    idx = [key for key, val in enumerate(hour_list) if val in overpass_time]
    print(idx)
    rel_hum = rel_hum[idx, :, :, :]
    print(rel_hum.shape)
    temp = temp[idx, :, :, :]
    
    def duplicates(lst, item):
        return [i for i, x in enumerate(lst) if x == item]
    
    remove_dup = dict((x, duplicates(overpass_time, x))[1] for x in set(overpass_time) 
                      if overpass_time.count(x) > 1)
    
    if len(idx) != len(calipso):
        cirrus_cover = np.delete(calipso, list(remove_dup.values()), axis = 0)
    else:
        cirrus_cover = calipso
        
    # transpose 2nd and 3rd axes (lat and lon) so that it matches the cirrus cover array and match size
    rel_hum = np.transpose(rel_hum, (0, 1, 3, 2))[:, :, :-1, :-1] 
    temp = np.transpose(temp, (0, 1, 3, 2))[:, :, :-1, :-1]
    rel_hum = np.where(np.isnan(cirrus_cover) == True, np.nan, rel_hum)
    temp = np.where(np.isnan(cirrus_cover) == True, np.nan, temp)
    
    # flatten the arrays (1D)
    cirrus_cover = cirrus_cover.reshape(-1)
    rel_hum = rel_hum.reshape(-1)
    temp = temp.reshape(-1)
    pres_lvl = np.tile(np.repeat(np.arange(100, 450, 50), np.repeat(20000, 7)), len(idx))
    
    # remove nan instances
    pres_lvl = pres_lvl[~(np.isnan(cirrus_cover))]
    cirrus_cover = cirrus_cover[~(np.isnan(cirrus_cover))] 
    rel_hum = rel_hum[~(np.isnan(rel_hum))] 
    temp = temp[~(np.isnan(temp))]
    
    corr = np.corrcoef(rel_hum, temp) # correlation between rel. hum. and temp
    
    X = np.column_stack((rel_hum, temp, pres_lvl))
    
    y = np.where(cirrus_cover > 0, 1, 0) # convert cirrus cover into binary response (cirrus or no cirrus) with certain thr

    return X, y

# X_train, y_train = model_arrays(path + "ERA5_data\\ERA5_jan15_train.nc", "LIDAR_01_15",
#                                 "2015-01-01 00:00", "2015-01-31 23:00", 744)

#%%
# testdata
X_test, y_test = model_arrays(path + "ERA5_data\\ERA5_feb15_test.nc", "LIDAR_02_15",
                              "2015-02-01 00:00", "2015-02-28 23:00", 672)

model.score(X_test, y_test)

#%%
'''
------------------------------RANDOM FOREST------------------------------------
'''

# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 100, random_state = 42, 
                            class_weight="balanced_subsample", ccp_alpha = 0.005)

# Train the model on training data
rf.fit(X_train[:,0:2], y_train)

# Use the forest's predict method on the test data
predictions = rf.predict(X_train[:,0:2])

threshold = 0.7

predicted_proba = rf.predict_proba(X_train[:,0:2])
predicted = (predicted_proba [:,1] >= threshold).astype('int')

# # define evaluation procedure
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# # evaluate model
# scores = cross_val_score(rf, X[:,0:2], y, scoring='roc_auc', cv=cv, n_jobs=-1)

#%%
cm = confusion_matrix(y_train, predicted)

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

#%%
# logreg model (https://realpython.com/logistic-regression-python/)
model = LogisticRegression(class_weight = 'balanced').fit(X_train, y_train)

# evaluate model
model.classes_
model.predict_proba(X_train) # 1st col is probability of predicted output being zero, second = 1 - first
model.predict(X_train) # class predictions
model.score(X_train, y_train)

#%%
cm = confusion_matrix(y_train, model.predict(X_train))

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