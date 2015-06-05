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

clf = ensemble.GradientBoostingRegressor(n_estimators=200, max_depth=3)
clf.fit(train[features], train['log-count'])
result = clf.predict(test[features])
result = np.expm1(result)

df = pd.DataFrame({'datetime':test['datetime'], 'count':result})
df.to_csv('results1.csv', index= False, columns=['datetime','count'])



