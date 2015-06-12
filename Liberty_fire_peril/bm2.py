"""
Beating the benchmark for Liberty Mutual Fund @ Kaggle

__author__ : Abhishek Thakur
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
#from sklearn import svm
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error
#from datetime import datetime
import datetime

start=datetime.datetime.now()
print ("Reading Training file....")
train = pd.read_csv('ttrain.csv')
print (datetime.datetime.now()-start)
print ("Reading Testing file....")
test = pd.read_csv('ttest.csv')
print (datetime.datetime.now()-start)
print ("Reading Target file....")
sample = pd.read_csv('sampleSubmission.csv')
print (datetime.datetime.now()-start)

tr = train[['var13','var15']]
ts = test[['var13','var15']]

print ("Preprocessing....")
## Pre-processing ##
tr = tr.fillna(0)
ts = ts.fillna(0)

#pca = PCA(n_components=6)
#pca.fit(tr)
#tr=pca.transform(tr)
#ts=pca.transform(ts)

print (ts.columns.tolist())
#scaler = preprocessing.MinMaxScaler()
#tr = pd.DataFrame(scaler.fit_transform(tr), columns=tr.columns)
#ts = pd.DataFrame(scaler.fit_transform(ts), columns=tr.columns)

tr = np.nan_to_num(np.array(tr))
ts = np.nan_to_num(np.array(ts))
print (datetime.datetime.now()-start)
print ("Builing Model....")
#clf = Ridge()
#clf = RandomForestRegressor (n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1,bootstrap=True, oob_score=False, n_jobs=1)
#clf = SVR(kernel='rbf', C=1e3, gamma=0.1)
clf = SVR(kernel='linear', C=1e3)
#clf = SVR(kernel='poly', C=1e3, degree=2)
#clf = GradientBoostingRegressor(n_estimators=100, learning_rate=0.2,max_depth=2, random_state=0, loss='ls')
clf.fit(tr, train['target'].values)
print (datetime.datetime.now()-start)
print ("Applying Model on Test data....")
preds = clf.predict(ts)
scores = cross_val_score(clf, tr, train['target'].values) 
mse = mean_squared_error(train['target'].values, clf.predict(tr))
print "Cross Validation Score:",scores.mean() 		  
print "Mean squared error",mse

f=open("cvscore.csv","a")
print >> f,scores.mean(),mse
f.close()
print (datetime.datetime.now()-start)
sample['target'] = preds
print ("Creating submission file....")
sample.to_csv('submission2.csv', index = False)
print (datetime.datetime.now()-start)

#print datetime.now()-start


