
# coding: utf-8

# In[23]:

import pandas as pd
import numpy as np
import json
import os
import seaborn as sns
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt

from collections import defaultdict


# In[24]:

np.random.seed(47)


# In[25]:

parent_dir = 'Desktop/'
rev =pd.read_csv(parent_dir +"review_cleaned.csv")


# In[26]:

rev.info()


# In[27]:

colsb = ['Unnamed: 0']
rev= rev.drop(colsb, axis=1)
rev.head()


# In[28]:

rev['text']=rev['text'].apply(lambda x:str(x).replace('b',''))
rev['text']=rev['text'].apply(lambda x:str(x).replace('\'',''))
rev.head()


# In[29]:

#Get the distribution of the ratings
x=rev['stars'].value_counts()
x=x.sort_index()
#plot
plt.figure(figsize=(8,4))
ax= sns.barplot(x.index, x.values, alpha=0.8)
plt.title("Star Rating Distribution")
plt.ylabel('# of reviews', fontsize=12)
plt.xlabel('Star Ratings ', fontsize=12)

#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()


# In[30]:

documents = [(t, star) for t,star in zip(rev['text'], rev['stars'])]
print(documents[0])


# In[31]:

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

#test
ps = PorterStemmer()
example_words = ["python","pythoner","pythoning","pythoned","pythonly"]

for w in example_words:
    print(ps.stem(w))


# In[32]:

len(documents)


# In[33]:

training_set = documents[:3000]
testing_set = documents[3000:]


# In[34]:

all_words = []

for (t, star) in training_set:
    for word in t.split():
        w = word.lower().replace('.', '').replace(',', '').replace('!', '')
        all_words.append(ps.stem(w))
        
all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15))
print("stupid appeared: " + str(all_words['stupid']) + "times")


# In[35]:

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


# In[36]:

len(featuresets)


# In[37]:

# POS to NEG encode
temp = []
pos_count = 0
for i in range(len(featuresets)):
    if featuresets[i][1] >= 4:
        temp.append((featuresets[i][0], 'pos'))
        pos_count = pos_count + 1
    elif featuresets[i][1] <=2:
        temp.append((featuresets[i][0], 'neg'))
print("pos rate: ", pos_count / len(temp))
print("len of temp: ", len(temp))


# In[38]:

training_set  = temp[:1500]
testing_set = temp[1500:]


# In[39]:

clf = nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bayes Classifier accuracy: ", nltk.classify.accuracy(clf, testing_set) * 100)
clf.show_most_informative_features(50)


# In[40]:

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier


Naive_Bayes_clf = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(Naive_Bayes_clf, testing_set))*100)
Naive_Bayes_clf.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MultinomialNB accuracy percent:",nltk.classify.accuracy(MNB_classifier, testing_set))

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BernoulliNB accuracy percent:",nltk.classify.accuracy(BNB_classifier, testing_set))

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

