import os
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import cross_validation
#from sklearn import pipeline, metrics, grid_search
#from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, \
                delimiter="\t", quoting=3)
test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", \
               quoting=3 )
y = train["sentiment"]  
print "Cleaning and parsing movie reviews...\n"      
traindata = []
for i in xrange( 0, len(train["review"])):
    traindata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], False)))
testdata = []
for i in xrange(0,len(test["review"])):import os
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import cross_validation
#from sklearn import pipeline, metrics, grid_search
#from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, \
                delimiter="\t", quoting=3)
test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", \
               quoting=3 )
y = train["sentiment"]  
print "Cleaning and parsing movie reviews...\n"      
traindata = []
for i in xrange( 0, len(train["review"])):
    traindata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], False)))
testdata = []
for i in xrange(0,len(test["review"])):
    testdata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], False)))
print 'vectorizing... ', 
tfv = TfidfVectorizer(min_df=3,  max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words = 'english')
X_all = traindata + testdata
lentrain = len(traindata)

print "fitting pipeline... ",
tfv.fit(X_all)
X_all = tfv.transform(X_all)

X = X_all[:lentrain]
X_test = X_all[lentrain:]

## Initialize the standard scaler 
#scl = StandardScaler()

# We will use SVM here..
#svm_model = SVC()

# Create the pipeline 
#clf = pipeline.Pipeline([('svm', svm_model)])

# Create a parameter grid to search for best parameters for everything in the pipeline
#param_grid = {'svm__C': [0.1,1,3],'svm__kernel': ['rbf','linear','poly'],'svm__gamma':[0,0.01,0.1]}

# Initialize Grid Search Model
#model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, 
#                                 verbose=10, n_jobs=-1, iid=True, refit=True, cv=5)
#
#
## Fit Grid Search Model
#model.fit(X, y)
#print("Best score: %0.3f" % model.best_score_)
#print("Best parameters set:")
#best_parameters = model.best_estimator_.get_params()
#for param_name in sorted(param_grid.keys()):
#	print("\t%s: %r" % (param_name, best_parameters[param_name]))
#
## Get best model
#best_model = model.best_estimator_


model1 = LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                         C=1, fit_intercept=True, intercept_scaling=1.0, 
                         class_weight=None, random_state=None)
model2 = SVC(C=1.0, kernel='rbf', degree=3, gamma=0.001, coef0=0.0, shrinking=True, 
            probability=False, tol=0.0001, cache_size=200, class_weight=None, verbose=False, 
            max_iter=-1, random_state=None)
model3 = SVC(C=1.0, kernel='linear', degree=3, gamma=0.001, coef0=0.0, shrinking=True, 
            probability=False, tol=0.0001, cache_size=200, class_weight=None, verbose=False, 
            max_iter=-1, random_state=None)


print "5 Fold CV Score from model1: ", np.mean(cross_validation.cross_val_score(model1, X, y, cv=5, scoring='roc_auc'))
print "5 Fold CV Score from model2: ", np.mean(cross_validation.cross_val_score(model2, X, y, cv=5, scoring='roc_auc'))
print "5 Fold CV Score from model3: ", np.mean(cross_validation.cross_val_score(model3, X, y, cv=5, scoring='roc_auc'))


print "Retrain on all training data, predicting test labels...\n"
model1.fit(X,y)
model2.fit(X,y)
model3.fit(X,y)
result1 = model1.predict_proba(X_test)[:,1]
result2 = model2.predict_proba(X_test)[:,1]
result3 = model3.predict_proba(X_test)[:,1]
testdata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], False)))
print 'vectorizing... ', 
tfv = TfidfVectorizer(min_df=3,  max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words = 'english')
X_all = traindata + testdata
lentrain = len(traindata)

print "fitting pipeline... ",
tfv.fit(X_all)
X_all = tfv.transform(X_all)

X = X_all[:lentrain]
X_test = X_all[lentrain:]

## Initialize the standard scaler 
#scl = StandardScaler()

# We will use SVM here..
#svm_model = SVC()

# Create the pipeline 
#clf = pipeline.Pipeline([('svm', svm_model)])

# Create a parameter grid to search for best parameters for everything in the pipeline
#param_grid = {'svm__C': [0.1,1,3],'svm__kernel': ['rbf','linear','poly'],'svm__gamma':[0,0.01,0.1]}

# Initialize Grid Search Model
#model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, 
#                                 verbose=10, n_jobs=-1, iid=True, refit=True, cv=5)
#
#
## Fit Grid Search Model
#model.fit(X, y)
#print("Best score: %0.3f" % model.best_score_)
#print("Best parameters set:")
#best_parameters = model.best_estimator_.get_params()
#for param_name in sorted(param_grid.keys()):
#	print("\t%s: %r" % (param_name, best_parameters[param_name]))
#
## Get best model
#best_model = model.best_estimator_


model1 = LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                         C=1, fit_intercept=True, intercept_scaling=1.0, 
                         class_weight=None, random_state=None)
model2 = SVC(C=1.0, kernel='rbf', degree=3, gamma=0.001, coef0=0.0, shrinking=True, 
            probability=False, tol=0.0001, cache_size=200, class_weight=None, verbose=False, 
            max_iter=-1, random_state=None)
model3 = SVC(C=1.0, kernel='linear', degree=3, gamma=0.001, coef0=0.0, shrinking=True, 
            probability=False, tol=0.0001, cache_size=200, class_weight=None, verbose=False, 
            max_iter=-1, random_state=None)


print "5 Fold CV Score from model1: ", np.mean(cross_validation.cross_val_score(model1, X, y, cv=5, scoring='roc_auc'))
print "5 Fold CV Score from model2: ", np.mean(cross_validation.cross_val_score(model2, X, y, cv=5, scoring='roc_auc'))
print "5 Fold CV Score from model3: ", np.mean(cross_validation.cross_val_score(model3, X, y, cv=5, scoring='roc_auc'))


print "Retrain on all training data, predicting test labels...\n"
model1.fit(X,y)
model2.fit(X,y)
model3.fit(X,y)
result1 = model1.predict_proba(X_test)[:,1]
result2 = model2.predict_proba(X_test)[:,1]
result3 = model3.predict_proba(X_test)[:,1]
result=result1*0.4+result2*0.3+result3*0.3
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'BWMBP_model.csv'), index=False, quoting=3)
print "Wrote results to BWMBP_model.csv"