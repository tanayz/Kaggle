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
from sklearn import pipeline
from sklearn.feature_extraction import text
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
import re
import string
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score,classification_report#,confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
# array declarations
sw=[]
s_data = []
s_labels = []
t_data = []
t_labels = []
stemmer = PorterStemmer()
#stopwords tweak - more overhead
stop_words = ['http','www','img','border','color','style','padding','table','font','thi','inch','ha','width','height',
'0','1','2','3','4','5','6','7','8','9']
#stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)

#stop_words = ['http','www','img','border','0','1','2','3','4','5','6','7','8','9']
stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)

punct = string.punctuation
punct_re = re.compile('[{}]'.format(re.escape(punct)))

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

# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)


if __name__ == '__main__':

    # Load the training file
    train = pd.read_csv("input/train.csv").fillna("")
    test  = pd.read_csv("input/test.csv").fillna("")

    # we dont need ID columns
    idx = test.id.values.astype(int)

    # create labels. drop useless columns
    y = train.median_relevance.values

    def preprocess(x):
        trq=[]        
        s=(" ").join(BeautifulSoup(x).get_text(" ").split(" ")) 
        s=re.sub("[^a-zA-Z]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ") if len(z)>1])
        s=re.sub("[^a-zA-Z]+"," ", s)
        s=" ".join([t for t in s.lower().split(" ") if t not in stop_words and len(t)>1])
        for token in s.split(' '):
            trq.append(token)
        return ' '.join(trq)
    # Fit TFIDF
    import scipy.sparse
    def vectorize(train, tfv_query=None):
        query_data = list(train['query'].apply(preprocess))
        title_data = list(train['product_title'].apply(preprocess))
        if tfv_query is None:
            tfv_query = TfidfVectorizer(min_df=3,  max_features=None,   
                    strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                    ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
                    stop_words =None)

            full_data = query_data + title_data
            tfv_query.fit(full_data)

        return scipy.sparse.hstack([tfv_query.transform(query_data), tfv_query.transform(title_data)]), tfv_query
    
    X, tfv_query = vectorize(train)
    X_test, _ = vectorize(test, tfv_query)
    
    # Initialize SVD
    svd = TruncatedSVD(n_components=400,algorithm='randomized')
    from sklearn.metrics.pairwise import linear_kernel
    class FeatureInserter():
        
        def __init__(self):
            pass
        
        def transform(self, X, y=None):
            distances = []
            quasi_jaccard = []
            print(len(distances), X.shape)
            
            for row in X.tocsr():
                row=row.toarray().ravel()
                cos_distance = linear_kernel(row[:row.shape[0]/2], row[row.shape[0]/2:])
                distances.append(cos_distance[0])
                intersect = row[:row.shape[0]/2].dot(row[row.shape[0]/2:])
                union = (row[:row.shape[0]/2]+row[row.shape[0]/2:]).dot((row[:row.shape[0]/2]+row[row.shape[0]/2:]))
                quasi_jaccard.append(1.0*intersect/union)
                
            print(len(distances), X.shape)
            print(distances[:10])
            
            #X = scipy.sparse.hstack([X, distances])
            return np.matrix([x for x in zip(distances, quasi_jaccard)])
            
        def fit(self, X,y):
            return self
            
        
        def fit_transform(self, X, y, **fit_params):
            self.fit(X,y)
            return self.transform(X)
    
    # Initialize the standard scaler 
    scl = StandardScaler()
    
    # We will use SVM here..
    svm_model = OneVsRestClassifier(SVC(C=10.))
    
    # Create the pipeline 
    model = pipeline.Pipeline([('UnionInput', FeatureUnion([('svd', svd), ('dense_features', FeatureInserter())])),('scl', scl)])
    
    model.fit(X,y)
    
    Xtrh=model.fit_transform(X)
    Xtsh=model.fit_transform(X_test)
    
        ############################# Transform data for h2o#############
    import h2o
    from h2o import H2OFrame     
    h2o.init()
    
    trd1=[]
    for l in Xtrh:
        trd1.append(l.tolist())
    
    trdy1=[]
    for k in range(len(y)):
        trdy1.append([y[k]])
        
    xtr1=H2OFrame(python_obj=trd1)
    ytr1=H2OFrame(python_obj=trdy1) 
    
    ytr1["C1"]._name = "C6001"  # Rename the default column
    
    tsd1=[]
    for l in Xtsh:
        tsd1.append(l.tolist())
    xts1=H2OFrame(python_obj=tsd1)
    
    
    ############################## Apply h2o models ####################
#    gb = h2o.gbm(x =xtr[1:175],y =ytr['C6001'],
#                distribution = "multinomial",
#                ntrees=1000, # 500 works well
#                max_depth=12,
#                learn_rate=0.01)
                
    dl1= h2o.deeplearning(x =xtr1[1:402],y =ytr1['C6001'],
                variable_importances=True,balance_classes=False,
                input_dropout_ratio=0.2,rho=0.99,
                hidden_dropout_ratios=[0.5,0.4,0.5,0.4],
                activation="Tanh",hidden=[402,600,402,4],epochs=75)
                
#    rf= h2o.random_forest(x =xtr[1:175],y =ytr['C6001'],
#                seed=1234, ntrees=600, 
#                max_depth=20, balance_classes=False)
    dlh1=dl1.predict(xtr1) 
    dly1=h2o.as_list(dlh1)
    yhat1=np.round(dly1['predict'].reshape(-1))
    yhat1=np.nan_to_num(yhat1)
    
    dls1 = dl1.predict(xts1)
    dls1=h2o.as_list(dls1)
    dls1=np.round(dls1['predict'].reshape(-1))
    dls1=np.nan_to_num(dls1)
    
    preds1=[]
    for d in dls1:
        preds1.append(dls1[d])


    #############################Pritint result######################
    yobs=y
    print 'Printing model performance'
#    print confusion_matrix(yobs, yhat),"\n",
    print accuracy_score(yobs,yhat1),"\n",classification_report(yobs,yhat1)    

    # Fit Model
#    model.fit(X, y)

#    preds = model.predict(X_test)    
    
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
    #create sklearn pipeline, fit all, and predict test data
    clf = Pipeline([('v',TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')), 
    ('svd', TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)), 
    ('scl', StandardScaler(copy=True, with_mean=True, with_std=True))]) 
#    ('svm', SVC(C=9.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None))])
#    clf.fit(s_data, s_labels)
    tr=clf.fit_transform(s_data)
    ts=clf.fit_transform(t_data)
    ############################# Transform data for h2o#############
    import h2o
    from h2o import H2OFrame     
    h2o.init(ip="localhost",strict_version_check=False)
    
    trd=[]
    for l in tr:
        trd.append(l.tolist())
    
    trdy=[]
    for k in range(len(s_labels)):
        trdy.append([s_labels[k]])
        
    xtr=H2OFrame(python_obj=trd)
    ytr=H2OFrame(python_obj=trdy) 
    
    ytr["C1"]._name = "C6001"  # Rename the default column
    
    tsd=[]
    for l in ts:
        tsd.append(l.tolist())
    xts=H2OFrame(python_obj=tsd)
    
    
    ############################## Apply h2o models ####################
#    gb = h2o.gbm(x =xtr[1:175],y =ytr['C6001'],
#                distribution = "multinomial",
#                ntrees=1000, # 500 works well
#                max_depth=12,
#                learn_rate=0.01)
                
    dl= h2o.deeplearning(x =xtr[1:200],y =ytr['C6001'],
                variable_importances=True,balance_classes=False,
                input_dropout_ratio=0.2,rho=0.99,
                hidden_dropout_ratios=[0.5,0.4,0.4,0.4],
                activation="Tanh",hidden=[200,400,200,4],epochs=80)
                
#    rf= h2o.random_forest(x =xtr[1:175],y =ytr['C6001'],
#                seed=1234, ntrees=600, 
#                max_depth=20, balance_classes=False)
    dlh=dl.predict(xtr) 
    dly=h2o.as_list(dlh)
    yhat2=np.round(dly['predict'].reshape(-1))

        
    
    #############################Pritint result######################
    yobs=y
    print 'Printing model performance'
#    print confusion_matrix(yobs, yhat),"\n",
    print accuracy_score(yobs,yhat2),"\n",classification_report(yobs,yhat2)    
    
    ################################################################
    dls2 = dl.predict(xts)
    dls2=h2o.as_list(dls2)
    dls2=np.round(dls2['predict'].reshape(-1))
    dls2=np.nan_to_num(dls2)
    
    preds2=[]
    for d in dls2:
        preds2.append(dls2[d])

#    t_labels = clf.predict(t_data)
    
    import math
    p3 = []
    for i in range(len(preds1)):
        x =math.sqrt(preds1[i]*preds2[i])
        x = round(x)
        p3.append(int(x))
    ##### Tell the performance of ensemble #######
#    yhat2=yhat2.astype(int)
    yobs=np.asarray(yobs,dtype=int)
    yhat=(yhat1+yhat2)/2
    yhat=np.round(yhat)

    print 'Ensemble by AM',accuracy_score(yobs,yhat),"\n",classification_report(yobs,yhat) 
    yhat=np.round(np.sqrt(yhat1*yhat2))
    yhat=yhat.astype(int)
#    yhat = round(yhat)
    print 'Ensemble by GM',accuracy_score(yobs,yhat),"\n",classification_report(yobs,yhat)    
#    print 'Model cv scores:',round(np.mean(scores1),4),round(np.mean(scores2),4)
    # Create your first submission file
    submission = pd.DataFrame({"id": idx, "prediction": p3})
    submission.to_csv("ensemble_svc_400_200_1st.csv", index=False)
