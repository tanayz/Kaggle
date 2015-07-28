import xgboost as xgb
#import numpy as np
#import pandas as pd
#from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
import numpy as np
#from prep import prep

#train,test,label_log,labels,idx=prep()

class Base(object):
    
    def __init__(self):
        self.vec = None
    
    def get_name(self):
        return self.__class__.__name__.lower()
    
    def get_regressor(self):
        params = {}
        params["objective"] = "reg:linear"
        params["eta"] = 0.05
        params["min_child_weight"] = 5
        params["subsample"] = 0.8
        params["colsample_bytree"] = 0.8
        params["scale_pos_weight"] = 1.0
        params["silent"] = 1
        params["max_depth"] = 7
        plst = list(params.items())
        num_rounds=np.random.randint(2000,8000)
        return plst,num_rounds
    
    def fit(self, x, labels):
        pass
    
    def predict(self, x):
        pass

#        return RandomForestRegressor(
#            n_estimators = np.random.randint(70,120),
#            max_depth = None,
#            n_jobs=-1,            
#            oob_score=True
#        )
#        return GradientBoostingRegressor(
#            n_estimators = np.random.randint(120,200),
#            max_depth = np.random.randint(6,10),
#            loss=['huber','ls', 'quantile'][np.random.randint(0,3)],            
#            verbose=10
#        )
