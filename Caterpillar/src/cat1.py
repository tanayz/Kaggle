import pandas as pd
import numpy as np
import glob
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer as DV
### Load train and test
test = pd.read_csv("../input/test_set.csv")
train = pd.read_csv("../input/train_set.csv")

train['id']=pd.Series(np.arange(-1,-train.shape[0],-1))
test['cost']=0

train=train.append(test)

continueLoop = True

while(continueLoop):
   continueLoop = False
   for f in glob.glob(os.path.join("../input/additionals/", '*.csv')):
        d=pd.read_csv(f)
        commonVariables=train.columns.difference(train.columns.difference(d.columns))
        if len(commonVariables)==1:
            train=pd.merge(train, d[d.columns.difference(train.columns)], left_index=True, right_index=True, how='outer')
            continueLoop=True
            print train.shape
        
from sklearn.base import TransformerMixin

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

test = train[train['id']>0]
train = train[train['id']<0]

rf = RandomForestRegressor(random_state=0, n_estimators=100)
y=np.log(train['cost']+1)
train1= train[train.columns.difference(train.columns[[2, 3]])]
test1=test[test.columns.difference(test.columns[[2, 3]])]
#rf.fit(X,y)

#### Separating numeric and character columns #####
numcols=[];charcols=[]
for i in range(len(train1.columns)):
    if train.ix[:,i].dtypes in ['int','float']:
        numcols.append(train1.columns[i])
    else:
        charcols.append(train1.columns[i])

x_num_train = train1[ numcols ].as_matrix()   
x_num_test = test1[ numcols ].as_matrix()
cat_train = train1[charcols]
cat_test = test1[charcols]

cat_train.fillna( 'NA', inplace = True )
cat_test.fillna( 'NA', inplace = True )

x_cat_train = cat_train.T.to_dict().values()
x_cat_test = cat_test.T.to_dict().values()
x_cat_test=x_cat_test+[x_cat_test[6]] # To retain row number to 30236

# vectorize

vectorizer = DV( sparse = False )
vec_x_cat_train = vectorizer.fit_transform( x_cat_train )
vec_x_cat_test = vectorizer.transform( x_cat_test )

# complete x

x_train = np.hstack(( x_num_train, vec_x_cat_train ))
x_test = np.hstack(( x_num_test, vec_x_cat_test ))


#### Train randomForest on the whole training set
#rf = randomForest(log(train$cost + 1)~., train[,-match(c("id", "cost"), names(train))], ntree = 30, do.trace = 2)
#
#pred = exp(predict(rf, test)) - 1
#
#submitDb = data.frame(id = test$id, cost = pred)
#submitDb = aggregate(data.frame(cost = submitDb$cost), by = list(id = submitDb$id), mean)
#
#write.csv(submitDb, "submit.csv", row.names = FALSE, quote = FALSE)


##### Clean NA values
#
#for i in range(len(train.columns)):
#    if train.ix[:,i].dtypes in ['int','float']:
#      train.ix[:,i].fillna(-1) 
#    else:
#       train.ix[:,i].fillna("NAvalue") 
#
##### Clean variables with too many categories

