# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 17:39:57 2015

@author: tanay
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn import pipeline, metrics, grid_search
    
    clf = Pipeline([('v',TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')), 
    ('svd', TruncatedSVD(n_components=450, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)), 
    ('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), 
    ('rf', RandomForestClassifier(n_estimators=20,max_depth=10))])


    param_grid = {'rf__n_estimators': [100,150],'rf__max_depth': [10,12]}
    
    model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, 
                                     verbose=10, n_jobs=-1, iid=True, refit=True, cv=5)
                                     
    # Fit Grid Search Model
    model.fit(s_data, s_labels)
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
    # Get best model
    best_model = model.best_estimator_
    
    # Fit model with best parameters optimized for quadratic_weighted_kappa
    best_model.fit(s_data, s_labels)
    preds = best_model.predict(t_data)
