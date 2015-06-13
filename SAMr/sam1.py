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

os.chdir("/home/tanay/Copy/Data/Kaggle/Sentiment_analysis/Data")

trf=pd.read_table("train.tsv")
tsf=pd.read_table("test.tsv")

#countvec = CountVectorizer()
#countvec.fit_transform(trf.Phrase)
#
#pd.DataFrame(countvec.fit_transform(trf.Phrase).toarray(), columns=countvec.get_feature_names())
'abc123def456ghi789zero0'.translate(None, digits)


