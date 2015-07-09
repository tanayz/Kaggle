import pandas as pd
import numpy as np
import glob
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
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


y=np.log(train['cost']+1)
train1= train[train.columns.difference(train.columns[[2, 3]])]
test1=test[test.columns.difference(test.columns[[2, 3]])]


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


### Clean variables with too many categories
for i in range(len(charcols)):
    print charcols[i],len(set(train1[charcols[i]]))
    tmp=train1[charcols[i]].value_counts()<30
    for j in range(len(train1[charcols[i]])):
        val=train1[charcols[i]][[j]].values[0]
        if val in tmp.index and tmp[val]:
            train1[charcols[i]][[j]]='rarevalue'
################################################


rf = RandomForestRegressor(random_state=0, n_estimators=30,max_depth=10)
rf.fit(train1,y)

yhat=rf.predict(train1)
print mean_squared_error(yhat,y)

pred=rf.predict(test1)

tmp=id.reshape(30235,1).astype('int')
tmp1=pred.reshape(30235,1)
tmp2=np.hstack([tmp,tmp1])
df=pd.DataFrame.from_records(tmp2,columns=['id','cost'])
df['id']=df['id'].astype(int)

df.to_csv("submission.csv",index=False)
