# -*- coding: utf-8 -*-
"""
@author: T Y van der Duim
"""

'''
------------------------------PACKAGES-----------------------------------------
'''

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
from CALIPSO import CALIPSO_analysis
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.dummy import DummyClassifier
import datetime 
from sklearn.metrics import precision_score, recall_score, accuracy_score
from miscellaneous import miscellaneous
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
---------------------------------PATHS----------------------------------------
'''

flight_path = 'E:\\Research_cirrus\\Flight_data\\'
lidar_path = 'E:\\Research_cirrus\\CALIPSO_data\\'
meteo_path = 'E:\\Research_cirrus\\ERA5_data\\'
meteosat_path = 'E:\\Research_cirrus\\Meteosat_CLAAS_data\\'

#%%

'''
-----------------------------LOGISTIC REGMODEL---------------------------------
'''

class ML_model:
    
    def __init__(self):
        self.model_arrays()

    def model_arrays(self):
    
        # import meteorological data
        data = miscellaneous.netcdf_import(meteo_path + 'ERA5_01_15.nc')
        rel_hum = data['r']
        temp = data['t']
        
        # get satellite data
        self.calipso = CALIPSO_analysis('01_15', time_res = 15, pressure_axis = True)
        self.dates = self.calipso.dates
        self.times = self.calipso.times
        self.temp_from_calipso = self.calipso.CALIPSO_temp
        
        # select parts of ERA5 data at times when CALIPSO overpasses Europe
        overpass_time = [datetime.datetime.combine(date, time) 
                          for date, time in zip(self.dates, self.times)] # merge dates and times to get a datetime list of Calipso overpasses
        
        hour_list = pd.date_range(start = datetime.datetime.strptime('2015-01-01 00:00', '%Y-%m-%d %H:%M'),
                                  end = datetime.datetime.strptime('2015-01-31 23:00', '%Y-%m-%d %H:%M'),
                                  periods = 744).to_pydatetime().tolist()
        
        overpass_time = [miscellaneous.hour_rounder(overpass) for overpass in overpass_time]
        idx = [key for key, val in enumerate(hour_list) if val in overpass_time]
        
        rel_hum = rel_hum[idx, :, :, :]
        temp = temp[idx, :, :, :]
        
        remove_dup = dict((x, miscellaneous.duplicates(overpass_time, x))[1] for x in set(overpass_time) 
                          if overpass_time.count(x) > 1)
        
        if len(idx) != len(self.calipso.CALIPSO_cirrus):
            cirrus_cover = np.delete(self.calipso.CALIPSO_cirrus, list(remove_dup.values()), axis = 0)[:, :, :, :]
        else:
            cirrus_cover = self.calipso.CALIPSO_cirrus[:, :, :, :]
            
        # transpose 2nd and 3rd axes (lat and lon) so that it matches the cirrus cover array and match size
        rel_hum = np.transpose(rel_hum, (0, 1, 3, 2))[:, :, :-1, :-1] 
        temp = np.transpose(temp, (0, 1, 3, 2))[:, :, :-1, :-1]
        rel_hum = np.where(np.isnan(cirrus_cover) == True, np.nan, rel_hum)
        temp = np.where(np.isnan(cirrus_cover) == True, np.nan, temp)
        
        try:
            temp_ERA5 = np.where(np.isnan(self.temp_from_calipso) == True, np.nan, temp)
        except:
            pass
        
        # flatten the arrays (1D)
        cirrus_cover = cirrus_cover.reshape(-1)
        rel_hum = rel_hum.reshape(-1)
        temp = temp.reshape(-1)
        pres_lvl = np.tile(np.repeat(np.array([100, 125, 150, 175, 200, 225,
            250, 300, 350, 400, 450]), np.repeat(20000, 11)), len(idx))
        temp_from_calipso = self.temp_from_calipso.reshape(-1)
        try:
            temp_ERA5 = temp_ERA5.reshape(-1)
        except:
            pass
        
        # remove nan instances
        pres_lvl = pres_lvl[~(np.isnan(cirrus_cover))]
        cirrus_cover = cirrus_cover[~(np.isnan(cirrus_cover))] 
        rel_hum = rel_hum[~(np.isnan(rel_hum))] 
        temp = temp[~(np.isnan(temp))]
        self.temp_from_calipso = temp_from_calipso[~(np.isnan(temp_from_calipso))]
        try:
            self.temp_ERA5 = temp_ERA5[~(np.isnan(temp_ERA5))]
        except:
            pass
                
        self.X = np.column_stack((rel_hum, temp, pres_lvl))
        
        self.y = np.where(cirrus_cover > 0, 1, 0) # convert cirrus cover into binary response (cirrus or no cirrus) with certain thr
        
        return self.X, self.y
    
    def compare_T_CAL_vs_ERA(self):
        self.temp_from_calipso = self.temp_from_calipso + 273.15
        
        temp_diff = np.subtract(self.temp_ERA5, self.temp_from_calipso)
        self.me = np.sqrt(np.square(temp_diff).mean())
        self.corr = round(np.corrcoef(self.temp_ERA5, self.temp_from_calipso)[0,1]**2,2)
        
        # plot
        fig, ax = plt.subplots(figsize=(8,8))
        lims = [200, 200,  # min of both axes
                250, 250,]  # max of both axes
        ax.scatter(self.temp_ERA5, self.temp_from_calipso, color = 'green', alpha = 0.2)
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=3, linestyle = 'dotted')
        ax.text(237, 245, '$R^2$ = ' + str(self.corr), fontsize = 24, weight = 'bold')
        ax.set_xlim([200, 250])
        ax.set_ylim([200, 250])
        ax.set_xlabel('Temperature from ERA5 (K)')
        ax.set_ylabel('Temperature from CALIPSO (K)')
    
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
        print(Counter(self.y_train_sampled))
        
    def logregmodel(self, threshold):
    
        model = LogisticRegression(class_weight = 'balanced').fit(self.X_train, self.y_train)
        
        accuracy = []
        precision = []
        
        for thr in np.arange(0, 1.1, 0.1):
            # evaluate model
            model.classes_
            model.predict(self.X_test) # class predictions
            model.predict_proba(self.X_test) # 1st col is probability of predicted output being zero, second = 1 - first
            predicted = (model.predict_proba(self.X_test) [:,1] >= thr).astype('int')
            cm = confusion_matrix(self.y_test, predicted)
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
            
        model.predict(self.X_test) # class predictions
        model.predict_proba(self.X_test) # 1st col is probability of predicted output being zero, second = 1 - first
        predicted = (model.predict_proba(self.X_test) [:,1] >= threshold).astype('int')
        cm = confusion_matrix(self.y_test, predicted)
            
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
        
        predicted_proba = self.rf.predict_proba(self.X_test)
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
    
    def RF_performanceplot(self, step):
        
        rf = RandomForestClassifier(n_estimators = 500, random_state = 42, 
                                                class_weight="balanced", oob_score = True)
        
        rf.fit(self.X_train, self.y_train)
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
                
        thresholds = np.arange(0.1, 1 + step, step)
        
        for thr in thresholds:
            print(thr)
            # evaluate model
            predicted_train = (rf.predict_proba(self.X_train)[:,1] >= thr).astype('int')
            predicted_test = (rf.predict_proba(self.X_test)[:,1] >= thr).astype('int')
            accuracy_train.append(accuracy_score(self.y_train, predicted_train))
            accuracy_test.append(accuracy_score(self.y_test, predicted_test))
            precision_train.append(precision_score(self.y_train, predicted_train))
            precision_test.append(precision_score(self.y_test, predicted_test))
            recall_train.append(recall_score(self.y_train, predicted_train))
            recall_test.append(recall_score(self.y_test, predicted_test))
        
        rf.fit(self.X_train_sampled, self.y_train_sampled)
        
        for thr in thresholds:
            # evaluate model
            predicted_train_sampled = (rf.predict_proba(self.X_train_sampled)[:,1] >= thr).astype('int')
            predicted_test_sampled = (rf.predict_proba(self.X_test)[:,1] >= thr).astype('int')
            accuracy_train_sampled.append(accuracy_score(self.y_train_sampled, predicted_train_sampled))
            accuracy_test_sampled.append(accuracy_score(self.y_test, predicted_test_sampled))
            precision_train_sampled.append(precision_score(self.y_train_sampled, predicted_train_sampled))
            precision_test_sampled.append(precision_score(self.y_test, predicted_test_sampled))
            recall_train_sampled.append(recall_score(self.y_train_sampled, predicted_train_sampled))
            recall_test_sampled.append(recall_score(self.y_test, predicted_test_sampled))
            
        # compute accuracy score in case of constant predictor (always predict no cirrus)
        dummy_clf = DummyClassifier(strategy="most_frequent")
        dummy_clf.fit(self.X_train, self.y_train)
        dummy_clf.predict(self.X_train)
        constant_pred = dummy_clf.score(self.X_train, self.y_train)
        
        dummy_clf.fit(self.X_train_sampled, self.y_train_sampled)
        dummy_clf.predict(self.X_train_sampled)
        constant_pred_sampled = dummy_clf.score(self.X_train_sampled, self.y_train_sampled)
        
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
        
    def LR_performance_plot(self):
        
        model = LogisticRegression(class_weight = 'balanced').fit(self.X_train, self.y_train)
        
        accuracy_test = []
        precision_test = []
        accuracy_train = []
        precision_train = []
        recall_test = []
        recall_train = []
        
        step = 0.05
        thr_range = np.arange(0, 1 + step, step)
        
        for thr in thr_range:
            # evaluate model
            predicted = (model.predict_proba(self.X_test) [:,1] >= thr).astype('int')
            cm = confusion_matrix(self.y_test, predicted)
            accuracy_test.append((cm[0,0] + cm[1, 1]) / np.sum(cm))
            precision_test.append(cm[1,1] / (cm[0,1] + cm[1,1]))
            recall_test.append(cm[1,1] / (cm[1,0] + cm[1,1]))
            
            predicted = (model.predict_proba(self.X_train) [:,1] >= thr).astype('int')
            cm = confusion_matrix(self.y_train, predicted)
            accuracy_train.append((cm[0,0] + cm[1, 1]) / np.sum(cm))
            precision_train.append(cm[1,1] / (cm[0,1] + cm[1,1]))
            recall_train.append(cm[1,1] / (cm[1,0] + cm[1,1]))
            
        accuracy = pd.DataFrame(list(zip(thr_range, accuracy_train, 
                                          accuracy_test)), columns = ['thresholds', 
                                          'accuracy_train', 
                                          'accuracy_test'])
        
        precision = pd.DataFrame(list(zip(thr_range, precision_train, 
                                          precision_test)), columns = ['thresholds', 
                                          'precision_train', 
                                          'precision_test'])
                                                                       
        recall = pd.DataFrame(list(zip(thr_range, recall_train, 
                                          recall_test)), columns = ['thresholds', 
                                          'recall_train', 
                                          'recall_test'])
        
        def plotting(metric, namevar):
            df = metric.melt("thresholds", var_name = namevar, value_name = 'score')
            df[namevar] = df[namevar].map(lambda x: x.lstrip('{0}_'.format(namevar)))
            df[['set']] = df[namevar].str.split('_', 1, expand=True)
            df = df.drop([namevar], axis = 1)
            return df
        
        accuracy = plotting(accuracy, 'accuracy')
        precision = plotting(precision, 'precision')
        recall = plotting(recall, 'recall')
        
        plt.rc('font', size=20) 
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=False, figsize=(12,8))
        sns.lineplot(ax = axes[0, 0], data = accuracy, x="thresholds", y="score", hue = 'set')
        axes[0, 0].set_title('accuracy')
        sns.lineplot(ax = axes[1, 1], data = precision, x="thresholds", y="score", hue = 'set')
        axes[1, 1].set_title('precision')
        sns.lineplot(ax = axes[1, 0], data = recall, x="thresholds", y="score", hue = 'set')
        axes[1, 0].set_title('recall')
        plt.tight_layout()
        plt.legend(bbox_to_anchor=(0.5, 1.7), borderaxespad=0.)
        axes[0, 0].legend([],[], frameon=False)
        axes[1, 0].legend([],[], frameon=False)
        axes[0, 1].xaxis.set_visible(False)
        axes[0, 1].yaxis.set_visible(False)
        axes[0, 0].axhline(0.92, color='red', ls='--')