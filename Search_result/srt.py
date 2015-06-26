# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 05:34:42 2015

@author: tanay
"""
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from bs4 import BeautifulSoup
import gensim
from gensim import corpora, models, similarities
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction import text
import pandas as pd
import numpy as np


# array declarations
sw=[]
trp = [];trq=[]
s_labels = []
tsp = [];tsq=[]
t_labels = []
#stopwords tweak - more overhead
stop_words = ['http','www','img','border','0','1','2','3','4','5','6','7','8','9','head']
stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)


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
    s=(" ").join([ z for z in BeautifulSoup(train["query"][i]).get_text(" ").split(" ")]) 
    s=re.sub("[^a-zA-Z]"," ", s)
    s= (" ").join([stemmer.stem(z) for z in s.split(" ") if len(z)>1])
    s=re.sub("[^a-zA-Z]+"," ", s)
    s=" ".join([x for x in s.lower().split(" ") if x not in stop_words and len(x)>1])
    trq.append(s)

   
for i in range(len(train.id)):
    s=(" ").join([ z for z in BeautifulSoup(train.product_title[i]).get_text(" ").split(" ")]) + " " + BeautifulSoup(train.product_description[i]).get_text(" ")
    s=re.sub("[^a-zA-Z]"," ", s)
    s= (" ").join([stemmer.stem(z) for z in s.split(" ") if len(z)>1])
    s=re.sub("[^a-zA-Z]+"," ", s)
    s=" ".join([x for x in s.lower().split(" ") if x not in stop_words and len(x)>1])
    trp.append(s)
    s_labels.append(str(train["median_relevance"][i]))
    
    
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

l1=[]  
l2=[int(x) for x in s_labels ] 
for i in range(10158):
    dat= corpora.Dictionary([trp[i].split(" ")]).doc2bow(trq[i].lower().split())
    l1.append(len(dat))

print np.corrcoef(l1, l2)[0, 1]
#    corpus = gensim.matutils.Dense2Corpus(np.array(dat))
#    dictionary=corpora.Dictionary([trp[i].split(" ")])
#    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2) 
#    print lsi[dat]
#    print dat,"\n",corpus
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    