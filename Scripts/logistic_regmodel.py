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
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.dummy import DummyClassifier

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
-----------------------------LOGISTIC REGMODEL---------------------------------
'''

path = 'E:\\Research_cirrus\\'

class ML_model:
    
    def __init__(self, file, calipso_folder, date_start, date_end, nr_periods): # option can be 'perform' or 'load' (interpolated data), savename its filename
        self.file = file
        self.calipso_folder = calipso_folder
        self.date_start = date_start
        self.date_end = date_end
        self.nr_periods = nr_periods
        self.model_arrays()
        #self.train_test_split()
        
    def netcdf_import(self):
        data = Dataset(self.file,'r')
        #print(data.variables)
        #print(data.variables.keys())
        my_dict = {}
        for key in data.variables.keys():
            if key == 'longitude' or key == 'latitude' or key == 'time' or key == 'level':
                my_dict[key] = data[key][:]
            else:
                my_dict[key] = data[key][:, :, :, :]
        return my_dict

    def model_arrays(self):
    
        data = self.netcdf_import()
        
        rel_hum = data['r']
        print(rel_hum.shape)
        temp = data['t']
        
        self.calipso = CALIPSO_analysis(path + "CALIPSO_data\\" + self.calipso_folder, time_res = 15, pressure_axis = True)
        # calipso_train = np.load('calipso_train.npy')
        # dates = np.load('dates_train.npy')
        # times = np.load('times_train.npy')
        temp_from_calipso = self.calipso.CALIPSO_temp
        
        # select parts of ERA5 data at times when CALIPSO overpasses Europe
        overpass_time = [datetime.combine(date, time) 
                          for date, time in zip(self.calipso.dates, self.calipso.times)] # merge dates and times to get a datetime list of Calipso overpasses
        print(overpass_time)
        def hour_rounder(t):
            # Rounds to nearest hour by adding a timedelta hour if minute >= 30
            return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
                       +timedelta(hours=t.minute//30))
        
        start = datetime.strptime(self.date_start, '%Y-%m-%d %H:%M')
        end = datetime.strptime(self.date_end, '%Y-%m-%d %H:%M')
        
        hour_list = pd.date_range(start = start, end = end, periods = self.nr_periods).to_pydatetime().tolist()
        
        overpass_time = [hour_rounder(overpass) for overpass in overpass_time]
        print(hour_list)
        idx = [key for key, val in enumerate(hour_list) if val in overpass_time]
        print(idx)
        rel_hum = rel_hum[idx, :, :, :]
        print(rel_hum.shape)
        temp = temp[idx, :, :, :]
        
        def duplicates(lst, item):
            return [i for i, x in enumerate(lst) if x == item]
        
        remove_dup = dict((x, duplicates(overpass_time, x))[1] for x in set(overpass_time) 
                          if overpass_time.count(x) > 1)
        
        if len(idx) != len(self.calipso.CALIPSO_cirrus):
            cirrus_cover = np.delete(self.calipso.CALIPSO_cirrus, list(remove_dup.values()), axis = 0)[:, 2:, :, :]
        else:
            cirrus_cover = self.calipso.CALIPSO_cirrus[:, 2:, :, :]
            
        # transpose 2nd and 3rd axes (lat and lon) so that it matches the cirrus cover array and match size
        rel_hum = np.transpose(rel_hum, (0, 1, 3, 2))[:, 2:, :-1, :-1] 
        temp = np.transpose(temp, (0, 1, 3, 2))[:, 2:, :-1, :-1]
        rel_hum = np.where(np.isnan(cirrus_cover) == True, np.nan, rel_hum)
        temp = np.where(np.isnan(cirrus_cover) == True, np.nan, temp)
        #temp_ERA5 = np.where(np.isnan(temp_from_calipso) == True, np.nan, temp)
        
        # flatten the arrays (1D)
        cirrus_cover = cirrus_cover.reshape(-1)
        rel_hum = rel_hum.reshape(-1)
        temp = temp.reshape(-1)
        pres_lvl = np.tile(np.repeat(np.array([150, 175, 200, 225,
            250, 300, 350, 400, 450]), np.repeat(20000, 9)), len(idx))
        #temp_from_calipso = calipso.CALIPSO_temp.reshape(-1)
        #temp_ERA5 = temp_ERA5.reshape(-1)
        
        # remove nan instances
        pres_lvl = pres_lvl[~(np.isnan(cirrus_cover))]
        cirrus_cover = cirrus_cover[~(np.isnan(cirrus_cover))] 
        rel_hum = rel_hum[~(np.isnan(rel_hum))] 
        temp = temp[~(np.isnan(temp))]
        #temp_from_calipso = temp_from_calipso[~(np.isnan(temp_from_calipso))]
        #temp_ERA5 = temp_ERA5[~(np.isnan(temp_ERA5))]
                
        self.X = np.column_stack((rel_hum, temp, pres_lvl))
        
        self.y = np.where(cirrus_cover > 0, 1, 0) # convert cirrus cover into binary response (cirrus or no cirrus) with certain thr
        
        return self.X, self.y
    
    def train_test_split(self):

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, 
                                                                                self.y, 
                                                                                test_size=0.5,
                                                                                random_state=42)
        
        # summarize class distribution
        print(Counter(self.y_train))
        
        # define undersample strategy
        oversample = RandomOverSampler(sampling_strategy=0.5)
        
        # fit and apply the transform
        self.X_train_sampled, self.y_train_sampled = oversample.fit_resample(self.X_train, self.y_train)
        print(Counter(self.y_train))
    
    def testdataplot(self):
        
        dataset = pd.DataFrame(list(zip(self.X_test[:, 0], self.X_test[:, 1], 
                                        self.X_test[:, 2], self.y_test)), columns = 
                       ['relative humidity', 'temperature (K)', 'pressure (hPa)', 'cirrus'])
        plt.figure()
        sns.pairplot(dataset, hue='cirrus', plot_kws={'alpha':0.1}, corner=True)
        fig = plt.gcf()
        fig.set_size_inches(15, 10)
        plt.show()
        
    def vary_depth_tree(self):
        error_train = []
        error_test = []
        depths = np.arange(0.1, 55, 5)
        
        for depth in depths:
            print(depth)
            # Instantiate model
            rf = RandomForestClassifier(n_estimators = 50, random_state = 42, 
                                        class_weight="balanced", oob_score = True,
                                        max_depth = depth)
            
            # Train the model on training data
            rf.fit(self.X_train, self.y_train)
            
            # Use the forest's predict method
            predictions_train = rf.predict(self.X_train)
            predictions_test = rf.predict(self.X_test)
            
            oob_error = 1 - rf.oob_score_
            error_train.append(oob_error)
            
            #cm_train = confusion_matrix(y_train, predictions_train)
            #error_train.append(1 - (cm_train[0,0] + cm_train[1, 1]) / np.sum(cm_train))
            
            cm_test = confusion_matrix(self.y_test, predictions_test)
            error_test.append(1 - (cm_test[0,0] + cm_test[1, 1]) / np.sum(cm_test))
        
        plt.figure()
        plt.plot(depths, error_train, color = 'b', label = 'training set')
        plt.plot(depths, error_test, color = 'r', label = 'test set')
        plt.xlabel("Max depth")
        plt.ylabel("Error")
        plt.title("RF performance on test data")
        plt.legend()
        plt.show()
        
        
    def confusion_matrix(self, threshold):
        
        predicted_proba = rf.predict_proba(self.X_test)
        predicted = (predicted_proba [:,1] >= threshold).astype('int')
        
        cm = confusion_matrix(self.y_test, predicted)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(cm, cmap = plt.cm.viridis)
        ax.grid(False)
        ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
        ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
        ax.set_ylim(1.5, -0.5)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
        plt.show()
    
    
rfmodel = ML_model(path + "ERA5_data\\ERA5_01_15.nc", "LIDAR_01_15",
                                "2015-01-01 00:00", "2015-01-31 23:00", 744)


#%%        
from sklearn.metrics import precision_score, recall_score, accuracy_score

rf = RandomForestClassifier(n_estimators = 100, random_state = 42, 
                                        class_weight="balanced", oob_score = True)

rf.fit(rfmodel.X_train, rfmodel.y_train)
accuracy_train = []
accuracy_train_sampled = []
accuracy_test = []
accuracy_test_sampled = []
precision_train = []
precision_train_sampled = []
precision_test = []
precision_test_sampled = []
recall_train = []
recall_train_sampled = []
recall_test = []
recall_test_sampled = []

#precision = []

thresholds = np.arange(0.1, 1.1, 0.1)

for thr in thresholds:
    print(thr)
    # evaluate model
    predicted_train = (rf.predict_proba(rfmodel.X_train)[:,1] >= thr).astype('int')
    predicted_test = (rf.predict_proba(rfmodel.X_test)[:,1] >= thr).astype('int')
    accuracy_train.append(accuracy_score(rfmodel.y_train, predicted_train))
    accuracy_test.append(accuracy_score(rfmodel.y_test, predicted_test))
    precision_train.append(precision_score(rfmodel.y_train, predicted_train))
    precision_test.append(precision_score(rfmodel.y_test, predicted_test))
    recall_train.append(recall_score(rfmodel.y_train, predicted_train))
    recall_test.append(recall_score(rfmodel.y_test, predicted_test))

rf.fit(rfmodel.X_train_sampled, rfmodel.y_train_sampled)

for thr in thresholds:
    print(thr)
    # evaluate model
    predicted_train_sampled = (rf.predict_proba(rfmodel.X_train_sampled)[:,1] >= thr).astype('int')
    predicted_test_sampled = (rf.predict_proba(rfmodel.X_test)[:,1] >= thr).astype('int')
    accuracy_train_sampled.append(accuracy_score(rfmodel.y_train_sampled, predicted_train_sampled))
    accuracy_test_sampled.append(accuracy_score(rfmodel.y_test, predicted_test_sampled))
    precision_train_sampled.append(precision_score(rfmodel.y_train_sampled, predicted_train_sampled))
    precision_test_sampled.append(precision_score(rfmodel.y_test, predicted_test_sampled))
    recall_train_sampled.append(recall_score(rfmodel.y_train_sampled, predicted_train_sampled))
    recall_test_sampled.append(recall_score(rfmodel.y_test, predicted_test_sampled))

#%%
# compute accuracy score in case of constant predictor (always predict no cirrus)
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(rfmodel.X_train, rfmodel.y_train)
dummy_clf.predict(rfmodel.X_train)
constant_pred = dummy_clf.score(rfmodel.X_train, rfmodel.y_train)

dummy_clf.fit(rfmodel.X_train_sampled, rfmodel.y_train_sampled)
dummy_clf.predict(rfmodel.X_train_sampled)
constant_pred_sampled = dummy_clf.score(rfmodel.X_train_sampled, rfmodel.y_train_sampled)

accuracy = pd.DataFrame(list(zip(thresholds, accuracy_train, accuracy_train_sampled, 
                                  accuracy_test, accuracy_test_sampled)), columns = ['thresholds', 
                                  'accuracy_train', 'accuracy_train_sampled', 
                                  'accuracy_test', 'accuracy_test_sampled'])

precision = pd.DataFrame(list(zip(thresholds, precision_train, precision_train_sampled, 
                                  precision_test, precision_test_sampled)), columns = ['thresholds', 
                                  'precision_train', 'precision_train_sampled', 
                                  'precision_test', 'precision_test_sampled'])
                                                                                    
recall = pd.DataFrame(list(zip(thresholds, recall_train, recall_train_sampled, 
                                  recall_test, recall_test_sampled)), columns = ['thresholds', 
                                  'recall_train', 'recall_train_sampled', 
                                  'recall_test', 'recall_test_sampled'])

def plotting(metric, namevar):
    df = metric.melt("thresholds", var_name = namevar, value_name = 'score')
    df[namevar] = df[namevar].map(lambda x: x.lstrip('{0}_'.format(namevar)))
    df[['set', 'oversampled']] = df[namevar].str.split('_', 1, expand=True)
    df = df.drop([namevar], axis = 1)
    df['oversampled'] = np.where(df['oversampled'] == 'sampled', 'yes', 'no')
    return df

accuracy = plotting(accuracy, 'accuracy')
precision = plotting(precision, 'precision')
recall = plotting(recall, 'recall')

plt.rc('font', size=16) 
fig, axes = plt.subplots(2, 2, sharex=True, sharey=False, figsize=(12,8))
fig.suptitle('Random Forest performance')
sns.lineplot(ax = axes[0, 0], data = accuracy, x="thresholds", y="score", hue = 'set', 
                  style = 'oversampled')
axes[0, 0].set_title('accuracy')
sns.lineplot(ax = axes[1, 1], data = precision, x="thresholds", y="score", hue = 'set', 
              style = 'oversampled')
axes[1, 1].set_title('precision')
sns.lineplot(ax = axes[1, 0], data = recall, x="thresholds", y="score", hue = 'set', 
              style = 'oversampled')
axes[1, 0].set_title('recall')
plt.tight_layout()
plt.legend(bbox_to_anchor=(0.3, 1.3), borderaxespad=0.)
axes[0, 0].legend([],[], frameon=False)
axes[1, 0].legend([],[], frameon=False)
axes[0, 1].xaxis.set_visible(False)
axes[0, 1].yaxis.set_visible(False)
axes[0, 0].axhline(constant_pred, color='red')
axes[0, 0].axhline(constant_pred_sampled, color='red', ls='--')
#axes[1, 1].legend([],[], frameon=False)

#%%

plt.figure()
plt.plot(thresholds, accuracy_train, color = 'b', label = 'accuracy train set')
plt.plot(thresholds, accuracy_train_sampled, color = 'r', label = 'accuracy train set (oversampling)')
plt.plot(thresholds, accuracy_test, color = 'g', label = 'accuracy test set')
plt.plot(thresholds, accuracy_test_sampled, color = 'darkorange', label = 'accuracy test set (oversampling)')
plt.plot(thresholds, precision_train, color = 'b', linestyle = 'dashed', label = 'precision train set')
plt.plot(thresholds, precision_train_sampled, color = 'r', linestyle = 'dashed', label = 'precision train set (oversampling)')
plt.plot(thresholds, precision_test, color = 'g', linestyle = 'dashed', label = 'precision test set')
plt.plot(thresholds, precision_test_sampled, color = 'darkorange', linestyle = 'dashed', label = 'precision test set (oversampling)')
plt.axhline(y = (cm[0, 0] + cm[0, 1]) / np.sum(cm), color = 'k', linestyle = 'dashed',
            label = 'constant predictor line')
plt.xlabel("Cut-off probability")
plt.ylabel("Score")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("Random Forest performance")
plt.legend()
plt.show()
#%%
# testdata
#X_test, y_test = model_arrays(path + "ERA5_data\\ERA5_feb15_test.nc", "LIDAR_02_15",
#                              "2015-02-01 00:00", "2015-02-28 23:00", 672)

#%%
# temp_ERA5 = temp_ERA5_test - 273.15 # convert from K to deg C

# # compute mse
# temp_diff = np.subtract(temp_ERA5, temp_calipso_test)
# mse = np.sqrt(np.square(temp_diff).mean())
# corr = round(np.corrcoef(temp_ERA5, temp_calipso_test)[0,1]**2,2)
# # plot
# fig, ax = plt.subplots(figsize=(8,8))
# lims = [-70, -70,  # min of both axes
#         -25, -25,]  # max of both axes

# # now plot both limits against eachother
# ax.scatter(temp_ERA5, temp_calipso_test, alpha = 0.5)
# ax.plot(lims, lims, 'k-', alpha=0.75, zorder=3)
# ax.text(-36, -28, '$r^2$ = ' + str(corr), fontsize = 24)
# ax.set_xlim([-70, -25])
# ax.set_ylim([-70, -25])
# ax.set_xlabel('Temperature from ERA5 ($^{\circ}$C)')
# ax.set_ylabel('Temperature from CALIPSO ($^{\circ}$C)')
# ax.set_title('Mean error = ' + str(round(mse, 1)) + '$^{\circ}$C')
# plt.savefig('temperatureERA5vsCALIPSO.png')

#%%
'''
------------------------------RANDOM FOREST------------------------------------
'''

#%%

feature_imp = pd.Series(rf.feature_importances_,index=['relative humidity', 'temperature',
                                                       'pressure level']).sort_values(ascending=False)

plt.figure()
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


#%%
'''
--------------------------LOGISTIC REGRESSION----------------------------------
'''
# (https://realpython.com/logistic-regression-python/)

def logregmodel(X_train, y_train, X_test, y_test, threshold):
    
    model = LogisticRegression(class_weight = 'balanced').fit(X_train, y_train)
    
    accuracy = []
    precision = []
    
    for thr in np.arange(0, 1.1, 0.1):
        # evaluate model
        model.classes_
        model.predict(X_test) # class predictions
        model.predict_proba(X_test) # 1st col is probability of predicted output being zero, second = 1 - first
        predicted = (model.predict_proba(X_test) [:,1] >= thr).astype('int')
        cm = confusion_matrix(y_test, predicted)
        accuracy.append((cm[0,0] + cm[1, 1]) / np.sum(cm))
        precision.append(cm[1,1] / (cm[0,1] + cm[1,1]))
        
    plt.figure()
    plt.plot(np.arange(0, 1.1, 0.1), accuracy, color = 'b', label = 'accuracy')
    plt.plot(np.arange(0, 1.1, 0.1), precision, color = 'r', label = 'precision')
    plt.axhline(y = (cm[0, 0] + cm[0, 1]) / np.sum(cm), color = 'k', linestyle = 'dashed',
                label = 'constant predictor line')
    plt.xlabel("Cut-off probability")
    plt.ylabel("Score")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("Logistic Regression performance on test data")
    plt.legend()
    plt.show()
        
    model.predict(X_test) # class predictions
    model.predict_proba(X_test) # 1st col is probability of predicted output being zero, second = 1 - first
    predicted = (model.predict_proba(X_test) [:,1] >= threshold).astype('int')
    cm = confusion_matrix(y_test, predicted)
        
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm, cmap = 'viridis')
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
    ax.set_title("Logreg confusion matrix on test data, thres = 0.8")
    plt.show()