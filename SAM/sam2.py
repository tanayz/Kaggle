# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 23:48:16 2014

@author: Tanay
"""

import os
import pandas as pd
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from string import digits
import string
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

os.chdir("/home/tanay/Copy/Data/Kaggle/SAM/data")

trf=pd.read_table("train.tsv")
tsf=pd.read_table("test.tsv")

bkp=trf
#countvec = CountVectorizer()
#countvec.fit_transform(trf.Phrase)
#
#pd.DataFrame(countvec.fit_transform(trf.Phrase).toarray(), columns=countvec.get_feature_names())
#'abc123def456ghi789zero0'.translate(None, digits)

for i in range(0,trf.Phrase.count()):
    trf.Phrase[i]=trf.Phrase[i].translate(None, digits)         #Remove numbers

table = string.maketrans("","")                                 #Remove punctuations

for i in range(0,trf.Phrase.count()):                           
    trf.Phrase[i]=trf.Phrase[i].translate(table, string.punctuation) 
    
#for i in range(1,trf.Phrase.count()):                          #Remove single s
#    trf.Phrase[i]=trf.Phrase[i].replace(' s ','') 
    
for i in range(0,trf.Phrase.count()):                           #Remove single word sentence
    trf.Phrase[i]=re.sub(r'\s[a-zA-Z]\s', ' ', trf.Phrase[i])    

for i in range(0,trf.Phrase.count()):                           #Remove single word sentence
    trf.Phrase[i]=re.sub(r'\b[a-zA-Z]\s', ' ',trf.Phrase[i])
 
for i in range(0,trf.Phrase.count()):                           #Remove single word sentence
    trf.Phrase[i]=re.sub(r'\s[a-zA-Z]\b', ' ',trf.Phrase[i])

for i in range(0,trf.Phrase.count()):                           #Remove double space
    trf.Phrase[i]=re.sub(r'\s[a-zA-Z]\b', ' ',trf.Phrase[i])

stop = stopwords.words('english')                              #Remove stopword
for j in range(0,trf.Phrase.count()):                                                              # remove stop words
   trf.Phrase[j]= ' '.join([i for i in trf.Phrase[j].split() if i not in stop])


for i in range(0,trf.Phrase.count()):                           #Stemming
  s=trf.Phrase[i].split()
  for j in range(0,len(s)):
    s[j]= str(SnowballStemmer("english").stem(s[j]))
  trf.Phrase[i]=' '.join(s)
                                                                # POS Tagging
for i in range(0,trf.Phrase.count()):                           # Only ADJ & ADV
  text = word_tokenize(trf.Phrase[i])
  temp=[]
  for k in range(0,len(text)-1):
   temp.append( ''.join(nltk.pos_tag(text)[k]))
  trf.Phrase[i]=' '.join(temp)
  
for i in range(0,trf.Phrase.count()):                           #Lowercase
    trf.Phrase[i]=trf.Phrase[i].lower()

#tfidf_vectorizer = TfidfVectorizer()
#tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_set)  


#finds the tfidf score with normalization  
  
                                                                
                                                                # Bag of words
                                                                # create word vector tf-idf
                                                                # Lowercase
                                                                # Unigram bigram
                                                                


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     