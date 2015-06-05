import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import math

from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.grid_search import GridSearchCV
from datetime import datetime

#Load Data with pandas, and parse the
#first column into datetime
train = pd.read_csv('data/train.csv', parse_dates=[0])
test = pd.read_csv('data/test.csv', parse_dates=[0])

#Feature engineering
temp = pd.DatetimeIndex(train['datetime'])
train['year'] = temp.year
train['month'] = temp.month
train['hour'] = temp.hour
train['weekday'] = temp.weekday

temp = pd.DatetimeIndex(test['datetime'])
test['year'] = temp.year
test['month'] = temp.month
test['hour'] = temp.hour
test['weekday'] = temp.weekday

#Define features vector
features = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'year', 'month', 'weekday', 'hour']


#the evaluation metric is the RMSE in the log domain,
#so we should transform the target columns into log domain as well.
for col in ['casual', 'registered', 'count']:
	train['log-' + col ] = train[col].apply(lambda x: np.log1p(x))



"""
clf = ensemble.GradientBoostingRegressor(n_estimators=200, max_depth=3)
clf.fit(train[features], train['log-count'])
result = clf.predict(test[features])
result = np.expm1(result)

df = pd.DataFrame({'datetime':test['datetime'], 'count':result})
df.to_csv('results1.csv', index= False, columns=['datetime','count'])
"""

#Split data into training and validation sets
temp = pd.DatetimeIndex(train['datetime'])
training = train[temp.day <= 16]
validation = train[temp.day > 16]

param_grid = {'learning_rate': [0.1, 0.05, 0.01], 'max_depth':[10,15,20], 'min_samples_leaf': [3,5,10,20]}
est = ensemble.GradientBoostingRegressor(n_estimators=500)

gs_cv = GridSearchCV(est, param_grid, n_jobs=4).fit(training[features],training['log-count'])

#best hyperparameter setting
gs_cv.best_params_

#Baseline error
error_count = mean_absolute_error(validation['log-count'], gs_cv.predict(validation[features]))

result = gs_cv.predict(test[features])
result = np.expm1(result)
df=pd.DataFrame({'datetime':test['datetime'], 'count':result})
df.to_csv('results2.csv', index = False, columns = ['datetime', 'count'])




