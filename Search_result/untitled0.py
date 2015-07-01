# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:05:47 2015

@author: uszllmd
"""
import pandas as pd
import numpy as np
#from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import pipeline, metrics, grid_search
#from sklearn import decomposition
from nltk.stem.porter import PorterStemmer
import re
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import text
from sklearn.metrics import accuracy_score,classification_report#,confusion_matrix

from numpy import genfromtxt

tr=genfromtxt("tr.csv",delimiter=',')

tr=tr.reshape(10158,180)

ts=genfromtxt("ts.csv",delimiter=',')

ts=ts.reshape(22513,180)

X=genfromtxt("X.csv",delimiter=',')

X=X.reshape(10158,305)

X_test=genfromtxt("X_test.csv",delimiter=',')

X_test=X_test.reshape(22513,305)

train = pd.read_csv("input/train.csv").fillna("")
test  = pd.read_csv("input/test.csv").fillna("")





train = pd.read_csv("input/train.csv").fillna("")
test  = pd.read_csv("input/test.csv").fillna("")


s_labels=[]

for i in range(len(train.id)):
    s_labels.append(str(train["median_relevance"][i]))