
# coding: utf-8

# In[5]:

import pandas as pd
import numpy as np
import json
import os
import seaborn as sns
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt


# In[4]:

parent_dir = 'Desktop/'
rev =pd.read_csv(parent_dir +"review_cleaned.csv")


# In[6]:

rev.info()


# In[7]:

colsb = ['Unnamed: 0']
rev= rev.drop(colsb, axis=1)
rev.head()


# In[8]:

rev['text']=rev['text'].apply(lambda x:str(x).replace('b',''))
rev['text']=rev['text'].apply(lambda x:str(x).replace('\'',''))
rev.head()


# In[9]:

documents = [(t, star) for t,star in zip(rev['text'], rev['stars'])]
print(documents[0])


# In[10]:

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

#test
ps = PorterStemmer()
example_words = ["python","pythoner","pythoning","pythoned","pythonly"]

for w in example_words:
    print(ps.stem(w))


# In[24]:

training_set = documents[:50000]
testing_set = documents[50000:]


# In[25]:

all_words = []

for (t, star) in training_set:
    for word in t.split():
        w = word.lower().replace('.', '').replace(',', '').replace('!', '')
        all_words.append(ps.stem(w))
        
all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15))
print("stupid appeared: " + str(all_words['stupid']) + "times")


# In[26]:

words_features = [s for (s,_) in list(all_words.most_common(3000))]

def find_features(document):
    words = set(document.split())
    features = {}
    for w in words:
        w = w.lower().replace('.', '').replace(',', '').replace('!', '')
        w = ps.stem(w)
        features[w] = (w in words_features)
    return features

featuresets = [(find_features(doc), star) for (doc,star) in training_set]
print(featuresets[0])


# In[29]:

training_set  = featuresets[:25000]
testing_set = featuresets[25000:]


# In[31]:

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

RandomForest_classifier = SklearnClassifier(RandomForestClassifier())
RandomForest_classifier.train(training_set)
print("RandomForest_classifier accuracy percet:", (nltk.classify.accuracy(RandomForest_classifier, testing_set))*100)

