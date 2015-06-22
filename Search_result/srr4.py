"""
Beating the Benchmark 
Search Results Relevance @ Kaggle
__Orginalauthor__ : Abhishek
__author__ : OverfitterScientist
"""
import pandas as pd
import numpy as np
#from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
#from sklearn import decomposition
from nltk.stem.porter import PorterStemmer
import re
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import text
from sklearn.metrics import accuracy_score,classification_report
from sklearn.ensemble import GradientBoostingClassifier

# array declarations
sw=[]
s_data = []
s_labels = []
t_data = []
t_labels = []
#stopwords tweak - more overhead
stop_words = ['http','www','img','border','0','1','2','3','4','5','6','7','8','9','head']
stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)
for stw in stop_words:
    sw.append("q"+stw)
    sw.append("z"+stw)
stop_words = text.ENGLISH_STOP_WORDS.union(sw)

if __name__ == '__main__':

    
    #load data
    train = pd.read_csv("input/train.csv").fillna("")
    test  = pd.read_csv("input/test.csv").fillna("")
    
    #remove html, remove non text or numeric, make query and title unique features for counts using prefix (accounted for in stopwords tweak)
    stemmer = PorterStemmer()
    ## Stemming functionality
    class stemmerUtility(object):
        """Stemming functionality"""
        @staticmethod
        def stemPorter(review_text):
            porter = PorterStemmer()
            preprocessed_docs = []
            for doc in review_text:
                final_doc = []
                for word in doc:
                    final_doc.append(porter.stem(word))
                    #final_doc.append(wordnet.lemmatize(word)) #note that lemmatize() can also takes part of speech as an argument!
                preprocessed_docs.append(final_doc)
            return preprocessed_docs
    
    
    for i in range(len(train.id)):
        s=(" ").join(["q"+ z for z in BeautifulSoup(train["query"][i]).get_text(" ").split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(train.product_title[i]).get_text(" ").split(" ")]) + " " + BeautifulSoup(train.product_description[i]).get_text(" ")
        s=re.sub("[^a-zA-Z0-9]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
        s_data.append(s)
        s_labels.append(str(train["median_relevance"][i]))
    for i in range(len(test.id)):
        s=(" ").join(["q"+ z for z in BeautifulSoup(test["query"][i]).get_text().split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(test.product_title[i]).get_text().split(" ")]) + " " + BeautifulSoup(test.product_description[i]).get_text()
        s=re.sub("[^a-zA-Z0-9]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
        t_data.append(s)
    #create sklearn pipeline, fit all, and predit test data
    clf = Pipeline([('v',TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')), 
    ('svd', TruncatedSVD(n_components=175, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)), 
    ('scl', StandardScaler(copy=True, with_mean=True, with_std=True))]) 
#    ('svm', SVC(C=9.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None))])
#    clf.fit(s_data, s_labels)
    tr=clf.fit_transform(s_data)
    ts=clf.fit_transform(t_data)
    
    clf = GradientBoostingClassifier(n_estimators=600, learning_rate=0.1,
     max_depth=10, random_state=0).fit(tr, s_labels)
     
    yobs=s_labels
    yhat=clf.predict(tr)
    ypred=clf.predict(ts)
    
    print 'Printing model performance'
#    print confusion_matrix(yobs, yhat),"\n",
    print accuracy_score(yobs,yhat),"\n",classification_report(yobs,yhat)    
#    from sklearn import cross_validation
#    scores = cross_validation.cross_val_score(clf, s_data, s_labels, cv=10,scoring='f1_weighted')
#    print round(np.mean(scores),4),"\n",scores
    
    ################################################################
#    t_labels = clf.predict(t_data)
    
    import math
    p3 = []
    for i in range(len(ypred)):
        x = (int(t_labels[i]) + ypred[i])/2
        x = math.floor(x)
        p3.append(int(x))
        
        
    ######################## Do the ensembling here ##################
    # p3 = (t_labels + preds)/2
    # p3 = p3.apply(lambda x:math.floor(x))
    # p3 = p3.apply(lambda x:int(x))
    
    # preds12 = 

    # Create your first submission file
    submission = pd.DataFrame({"id": idx, "prediction": p3})
    submission.to_csv("submmission_v6.csv", index=False)