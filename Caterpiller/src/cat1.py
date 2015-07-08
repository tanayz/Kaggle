import pandas as pd
import numpy as np
import glob
import os

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
        

#### Clean NA values
#for(i in 1:ncol(train)){
#  if(is.numeric(train[,i])){
#    train[is.na(train[,i]),i] = -1
#  }else{
#    train[,i] = as.character(train[,i])
#    train[is.na(train[,i]),i] = "NAvalue"
#    train[,i] = as.factor(train[,i])
#  }
#}
#
#### Clean variables with too many categories
#for(i in 1:ncol(train)){
#  if(!is.numeric(train[,i])){
#    freq = data.frame(table(train[,i]))
#    freq = freq[order(freq$Freq, decreasing = TRUE),]
#    train[,i] = as.character(match(train[,i], freq$Var1[1:30]))
#    train[is.na(train[,i]),i] = "rareValue"
#    train[,i] = as.factor(train[,i])
#  }
#}
#
#test = train[which(train$id > 0),]
#train = train[which(train$id < 0),]
