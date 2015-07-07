# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 06:37:25 2015

@author: tanay
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn import pipeline, metrics, grid_search

clf = pipeline.Pipeline([('rf', RandomForestClassifier())])



param_grid = {'rf__n_estimators': [140,160,180,200],'rf__max_depth': [12,14,16]}

model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, 
                                 verbose=10, n_jobs=-1, iid=True, refit=True, cv=5)
                                 
# Fit Grid Search Model
model.fit(Xtrh, y)
print("Best score: %0.3f" % model.best_score_)
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()

for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

#     rf__max_depth: 16         rf__n_estimators: 140,300
#	rf__n_estimators: 160,	svd__n_components: 300 and 0.63 came for part 2