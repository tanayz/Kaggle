"""
Author:Tanay Chowdhury
Following the R script from arnaud demytt 
"""
import pandas as pd
import numpy as np
import glob
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn import ensemble, preprocessing
import xgboost as xgb
from sklearn.base import TransformerMixin
from sklearn import pipeline, metrics, grid_search


### Load train and test
test = pd.read_csv("../input/test_set.csv")
train = pd.read_csv("../input/train_set.csv")

train['id']=pd.Series(np.arange(-1,-train.shape[0],-1))
test['cost']=0

train=train.append(test)

continueLoop = True

while(continueLoop):
   continueLoop = False
   for f in glob.glob(os.path.join("../input/", '*.csv')):
        d=pd.read_csv(f)
        commonVariables=train.columns.difference(train.columns.difference(d.columns))
        if len(commonVariables)==1:
            train=pd.merge(train, d[d.columns.difference(train.columns)], left_index=True, right_index=True, how='outer')
            continueLoop=True
            print train.shape

        

test = train[train['id']>0]
train = train[train['id']<0]

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

train = DataFrameImputer().fit_transform(train)
test = DataFrameImputer().fit_transform(test)

y=np.log1p(train.cost.values) #train.cost
train1= train[train.columns.difference(train.columns[[2, 3]])].copy()
test1=test[test.columns.difference(test.columns[[2, 3]])].copy()


#### Separating numeric and character columns #####
numcols=[];charcols=[]
for i in range(len(train1.columns)):
    if train1.ix[:,i].dtypes != 'O':
        numcols.append(train1.columns[i])
    else:
        charcols.append(train1.columns[i])

for i in range(len(train1.columns)):
    if train1.ix[:,i].dtypes  != 'O':
      train1.ix[:,i]=train1.ix[:,i].fillna(-1) 
    else:
       train1.ix[:,i]=train1.ix[:,i].fillna("NAvalue") 
       
for i in range(len(test1.columns)):
    if test1.ix[:,i].dtypes  != 'O':
      test1.ix[:,i]=test1.ix[:,i].fillna(-1) 
    else:
       test1.ix[:,i]=test1.ix[:,i].fillna("NAvalue") 

# convert categorical data to numpy array
ctrain1 = np.array(train1[charcols])
ctest1 = np.array(test1[charcols])


# label encode the categorical variables
for i in range(ctrain1.shape[1]):
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(ctrain1[:,i]) + list(ctest1[:,i]))
        ctrain1[:,i] = lbl.transform(ctrain1[:,i])
        ctest1[:,i] = lbl.transform(ctest1[:,i])

# convert numeric data to numpy array
ntrain1 = np.array(train1[numcols])
ntest1 = np.array(test1[numcols])

train2=np.hstack([ntrain1,ctrain1]).astype(float)
test2=np.hstack([ntest1,ctest1]).astype(float)

###############################Get best model##################
clf = RandomForestRegressor(n_estimators=50,max_depth=10,oob_score=True,n_jobs=-1)
clf.fit(train2,y)

yhat=clf.predict(train2)
print mean_squared_error(yhat,y)


#param_grid = {'C': [0.1,1,10],'gamma': [0,0.0001]}
#
#model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, 
#                                 verbose=10, n_jobs=-1, iid=True, refit=True, cv=5)
#                                 
##model = grid_search.RandomizedSearchCV(estimator = clf, param_distributions=param_grid, 
##                                 verbose=10, n_jobs=-1, cv=5)
#
#
## Fit Grid Search Model
#model.fit(train2,y)
#print("Best score: %0.3f" % model.best_score_)
#print("Best parameters set:")
#best_parameters = model.best_estimator_.get_params()
#
#for param_name in sorted(param_grid.keys()):
#    print("\t%s: %r" % (param_name, best_parameters[param_name]))
#RF best score 0.686 w/n=50,m=10


pred=clf.predict(test2)
pred=np.expm1(pred).reshape(30235,1)

id=test['id'].reshape(30235,1).astype('int')
res=np.hstack([id,pred])
df=pd.DataFrame.from_records(res,columns=['id','cost'])
df['id']=df['id'].astype(int)

df.to_csv("submission.csv",index=False)
