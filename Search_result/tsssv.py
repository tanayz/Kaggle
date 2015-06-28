# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 05:34:42 2015

@author: tanay
"""
from __future__ import division
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from bs4 import BeautifulSoup
from gensim import corpora
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction import text
import pandas as pd
#import numpy as np

def sensimvar():    
    # array declarations
#    sw=[]
#    trp = [];trq=[]
#    s_labels = []
    tsp = [];tsq=[]
#    t_labels = []
    #stopwords tweak - more overhead
    stop_words = ['http','www','img','border','0','1','2','3','4','5','6','7','8','9','head']
    stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)
    
    
    #load data
#    train = pd.read_csv("input/train.csv").fillna("")
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
        
#    for i in range(len(train.id)):
#        s=(" ").join([ z for z in BeautifulSoup(train["query"][i]).get_text(" ").split(" ")]) 
#        s=re.sub("[^a-zA-Z]"," ", s)
#        s= (" ").join([stemmer.stem(z) for z in s.split(" ") if len(z)>1])
#        s=re.sub("[^a-zA-Z]+"," ", s)
#        s=" ".join([x for x in s.lower().split(" ") if x not in stop_words and len(x)>1])
#        trq.append(s)
#    
#       
#    for i in range(len(train.id)):
#        s=(" ").join([ z for z in BeautifulSoup(train.product_title[i]).get_text(" ").split(" ")]) + " " + BeautifulSoup(train.product_description[i]).get_text(" ")
#        s=re.sub("[^a-zA-Z]"," ", s)
#        s= (" ").join([stemmer.stem(z) for z in s.split(" ") if len(z)>1])
#        s=re.sub("[^a-zA-Z]+"," ", s)
#        s=" ".join([x for x in s.lower().split(" ") if x not in stop_words and len(x)>1])
#        trp.append(s)
#        s_labels.append(str(train["median_relevance"][i]))
        
        
    for i in range(len(test.id)):
        s=(" ").join([ z for z in BeautifulSoup(test["query"][i]).get_text(" ").split(" ")]) 
        s=re.sub("[^a-zA-Z]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ") if len(z)>1])
        s=re.sub("[^a-zA-Z]+"," ", s)
        s=" ".join([x for x in s.lower().split(" ") if x not in stop_words and len(x)>1])
        tsq.append(s)
        
    
    for i in range(len(test.id)):
        s=(" ").join([ z for z in BeautifulSoup(test.product_title[i]).get_text().split(" ")]) + " " + BeautifulSoup(test.product_description[i]).get_text()
        s=re.sub("[^a-zA-Z]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ") if len(z)>1])
        s=re.sub("[^a-zA-Z]+"," ", s)
        s=" ".join([x for x in s.lower().split(" ") if x not in stop_words and len(x)>1])
        tsp.append(s.lower())
    
    l2=[]  
#    l1=[int(x) for x in s_labels ] 
    for i in range(len(test)):
        dat= corpora.Dictionary([tsp[i].split(" ")]).doc2bow(tsq[i].lower().split())
        l2.append(len(dat)/(len(dat)+1))
    
#    print np.corrcoef(l1, l2)[0, 1]
    
    
#    from collections import Counter
    
#    cnt=Counter(s.split(" "))
#    cnt.most_common()
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer() 
    
    tfv=[];lv=[]
    for i in range(10158):       
        tfs = tfidf.fit_transform(tsp[i].split(" "))
        response = tfidf.transform(tsq[i].split(" ")) 
    #    print response  
        val=0    
#        feature_names = tfidf.get_feature_names()
        for col in response.nonzero()[1]:
    #        print feature_names[col], ' - ', response[0, col]
            val=val+response[0, col]
        tfv.append(val)
        lv.append(len(response.nonzero()[0]))
        
#    print np.corrcoef(l1, tfv)[0, 1]
#    print np.corrcoef(l1, lv)[0, 1]     
        
        
    import sensim
    sf=[];st=[]
    
    for i in range(10158):
        sf.append(sensim.similarity(tsq[i], tsp[i], False))
        st.append(sensim.similarity(tsq[i], tsp[i], True))
    
    return [l2]+[tfv]+[lv]+[sf]+[st]
#    print np.corrcoef(l1, sf)[0, 1] #0.241178664079
#    print np.corrcoef(l1, st)[0, 1] #0.262722428745 
    
    #w/o stemming
    #0.235172215417
    #0.261363416018
    
    #sf=[];st=[]
    #from joblib import Parallel, delayed
    #sf=(Parallel(n_jobs=6)(delayed(sensim.similarity)(trq[i], trp[i], False) for i in range(10158)))
    #st=(Parallel(n_jobs=6)(delayed(sensim.similarity)(trq[j], trp[j], True) for j in range(10158)))




   
    
    
    
    
    