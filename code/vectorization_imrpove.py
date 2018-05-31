
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

parent_dir = 'Desktop/dataset/'
rev=pd.read_csv(parent_dir +"rev_new.csv")
cols = ['Unnamed: 0','business_id','useful','user_id','review_id']
rev= rev.drop(cols, axis=1)
rev=rev.dropna(subset = ['Rating'])
rev['text']=rev['text'].apply(lambda x:str(x).replace('\'','').replace('\\n',''))
rev['text']=rev['text'].apply(lambda x:str(x).replace('"',''))
rev.head()


# In[5]:

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

#test
ps = PorterStemmer()
example_words = ["python","pythoner","pythoning","pythoned","pythonly"]

for w in example_words:
    print(ps.stem(w))


# In[9]:

documents = [(t, star) for t,star in zip(rev['text'], rev['stars'])]
test = documents[:3000]
all_words = []

for (t, star) in test:
    for word in t.split():
        w = word.lower().replace('.', '').replace(',', '').replace('!', '').replace('"', '').replace('\n', '')
        all_words.append(ps.stem(w))
        
all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15))

words_features = [s for (s,_) in list(all_words.most_common(3000))]
def find_features(document):
    words = set(document.split())
    features = {}
    for w in words:
        w = w.lower().replace('.', '').replace(',', '').replace('!', '').replace('"', '').replace('\n', '').replace('\'', '').replace('b\'', '').replace('\\n', '')
        w = ps.stem(w)
        features[w] = (w in words_features)
    return features

featuresets = [(find_features(doc), star) for (doc,star) in test]

print(featuresets[0])

# POS to NEG encode
temp = []
pos_count = 0
for i in range(len(featuresets)):
    if featuresets[i][1] >= 4:
        temp.append((featuresets[i][0], 'pos'))
        pos_count = pos_count + 1
    elif featuresets[i][1] <= 2:
        temp.append((featuresets[i][0], 'neg'))
print("pos rate: ", pos_count / len(temp))
print("len of temp: ", len(temp))


# In[11]:

training_set  = temp[:5000]
testing_set = temp[5000:]

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

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

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


# In[39]:

rev.info()


# In[76]:

rev_new = rev[:981166]
rev_new.head()


# In[77]:

rev_new["Rating"] = rev_new["Rating"].astype(int)
rev_new.head()


# In[78]:

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(rev_new['text'], rev_new['Rating'], train_size = .75, random_state = 47)
print("Shape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)



# In[79]:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
vect = CountVectorizer(max_df = 0.95, min_df = 3, stop_words = 'english').fit(X_train)
vect.get_feature_names()[::2000]


# In[80]:

print("Total number of features: ", len(vect.get_feature_names()))


# In[81]:

# Now we vectorize the X_train data# Now we 
X_train_vectorized = vect.transform(X_train)
X_train_vectorized


# In[82]:

# Try using a lineaSVC
clf = LinearSVC()
clf.fit(X_train_vectorized, y_train)

pred = clf.predict(vect.transform(X_test))

print(confusion_matrix(y_true = y_test, y_pred = pred))
print("Test accuracy: ", (confusion_matrix(y_true = y_test, y_pred = pred)[0][0] + confusion_matrix(y_true = y_test, y_pred = pred)[1][1])/len(X_test))
print('AUC: ', roc_auc_score(y_test, pred))


# In[83]:

# Try using a logistic regression
clf2 = LogisticRegression()
clf2.fit(X_train_vectorized, y_train)

pred = clf2.predict(vect.transform(X_test))

print(confusion_matrix(y_true = y_test, y_pred = pred))
print("Test accuracy: ", (confusion_matrix(y_true = y_test, y_pred = pred)[0][0] + confusion_matrix(y_true = y_test, y_pred = pred)[1][1])/len(X_test))
print('AUC: ', roc_auc_score(y_test, pred))


# In[84]:

feature_names = np.array(vect.get_feature_names())

sored_coef_index = clf.coef_[0].argsort()
print("Smallest Coefs: \n{}\n".format(feature_names[sored_coef_index[:15]]))
print("Biggest Coefs: \n{}\n".format(feature_names[sored_coef_index[:-16:-1]]))


# In[85]:

vect = TfidfVectorizer(max_df = .95, min_df = 3, stop_words = 'english').fit(X_train)
len(vect.get_feature_names())


# In[86]:

X_train_vectorized = vect.transform(X_train)

clf = LinearSVC()
clf.fit(X_train_vectorized, y_train)

pred = clf.predict(vect.transform(X_test))

print(confusion_matrix(y_true = y_test, y_pred = pred))
print("Test accuracy: ", (confusion_matrix(y_true = y_test, y_pred = pred)[0][0] + confusion_matrix(y_true = y_test, y_pred = pred)[1][1])/len(X_test))
print('AUC: ', roc_auc_score(y_test, pred))


# In[87]:

feature_names = np.array(vect.get_feature_names())

sored_coef_index = clf.coef_[0].argsort()
print("Smallest Coefs: \n{}\n".format(feature_names[sored_coef_index[:25]]))
print("Biggest Coefs: \n{}\n".format(feature_names[sored_coef_index[:-26:-1]]))


# In[88]:

vect = TfidfVectorizer(max_df = .95, min_df = 3, ngram_range = (1,2)).fit(X_train)
print(len(vect.get_feature_names()))


# In[89]:

X_train_vectorized = vect.transform(X_train)

clf = LinearSVC()
clf.fit(X_train_vectorized, y_train)

pred = clf.predict(vect.transform(X_test))

print(confusion_matrix(y_true = y_test, y_pred = pred))
print("Test accuracy: ", (confusion_matrix(y_true = y_test, y_pred = pred)[0][0] + confusion_matrix(y_true = y_test, y_pred = pred)[1][1])/len(X_test))
print('AUC: ', roc_auc_score(y_test, pred))


# In[90]:

feature_names = np.array(vect.get_feature_names())

sored_coef_index = clf.coef_[0].argsort()
print("Smallest Coefs: \n{}\n".format(feature_names[sored_coef_index[:45]]))
print("Biggest Coefs: \n{}\n".format(feature_names[sored_coef_index[:-46:-1]]))


# In[91]:

# LogisticRegression
clf_LogisticRegression = LogisticRegression(class_weight = 'balanced') #class_weight = 'balanced'
clf_LogisticRegression.fit(X_train_vectorized, y_train)

pred = clf_LogisticRegression.predict(vect.transform(X_test))
print("LogisticRegression accuracy: ", (confusion_matrix(y_true = y_test, y_pred = pred)[0][0] + confusion_matrix(y_true = y_test, y_pred = pred)[1][1])/len(X_test))
print(confusion_matrix(y_true = y_test, y_pred = pred))
print('AUC: ', roc_auc_score(y_test, pred))

# LinearSVC
clf_LinearSVC = LinearSVC() #class_weight = 'balanced'
clf_LinearSVC.fit(X_train_vectorized, y_train)

pred = clf_LinearSVC.predict(vect.transform(X_test))
print("LinearSVC accuracy: ", (confusion_matrix(y_true = y_test, y_pred = pred)[0][0] + confusion_matrix(y_true = y_test, y_pred = pred)[1][1])/len(X_test))
print(confusion_matrix(y_true = y_test, y_pred = pred))
print('AUC: ', roc_auc_score(y_test, pred))

# SGDClassifier
clf_SGDClassifier = SGDClassifier(class_weight = 'balanced') #class_weight = 'balanced'
clf_SGDClassifier.fit(X_train_vectorized, y_train)

pred = clf_SGDClassifier.predict(vect.transform(X_test))
print("SGDClassifier accuracy: ", (confusion_matrix(y_true = y_test, y_pred = pred)[0][0] + confusion_matrix(y_true = y_test, y_pred = pred)[1][1])/len(X_test))
print(confusion_matrix(y_true = y_test, y_pred = pred))
print('AUC: ', roc_auc_score(y_test, pred))


# In[74]:

from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


# In[72]:

# Deafult values and helper function to print the topics
n_features = 5000
n_components = 10
n_top = 15

def print_top_words(model, feature_names, n_top):
    for topic_ind, topic in enumerate(model.components_):
        message = "Topic %d: " % topic_ind
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top - 1:-1]])
        print(message)
    print()

# Extract TF for LDA:
count_vect = CountVectorizer(max_df = .95, min_df = 3, ngram_range = (1,2),                              max_features = n_features, stop_words = 'english')
tf = count_vect.fit_transform(X_train)

# Extract TF-IDF for NMF:
tfidf_vect = TfidfVectorizer(max_df = .95, min_df = 3, ngram_range = (1,2),                              max_features = n_features, stop_words = 'english')
tfidf = tfidf_vect.fit_transform(X_train)
print('finished')


# In[96]:

# Fit the NMF model
nmf = NMF(n_components = 14, random_state = 47, alpha = .1, l1_ratio = .5).fit(tfidf)
print("\nTopics from NMF model: ")
tfidf_feature_names = tfidf_vect.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top)


# In[95]:

# Fit the LDA model
lda = LatentDirichletAllocation(n_topics = 14, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=47).fit(tf)
print_top_words(lda, tfidf_feature_names, n_top)


# In[97]:

# Run NMF
nmf = NMF(n_components=14, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

# Run LDA
lda = LatentDirichletAllocation(n_topics=14, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)




# In[100]:

X_train, X_test, y_train, y_test = train_test_split(rev_new['text'], rev_new['Rating'], train_size = .75, random_state = 47)


# In[99]:

X_train_vectorized = nmf.transform(tfidf)

clf = LinearSVC()
clf.fit(X_train_vectorized, y_train)

pred = clf.predict(nmf.transform(X_test))

print(confusion_matrix(y_true = y_test, y_pred = pred))
print("Test accuracy: ", (confusion_matrix(y_true = y_test, y_pred = pred)[0][0] + confusion_matrix(y_true = y_test, y_pred = pred)[1][1])/len(X_test))
print('AUC: ', roc_auc_score(y_test, pred))


# In[ ]:

X_train_vectorized = lda.transform(tf)

clf = LinearSVC()
clf.fit(X_train_vectorized, y_train)

pred = clf.predict(vect.transform(X_test))

print(confusion_matrix(y_true = y_test, y_pred = pred))
print("Test accuracy: ", (confusion_matrix(y_true = y_test, y_pred = pred)[0][0] + confusion_matrix(y_true = y_test, y_pred = pred)[1][1])/len(X_test))
print('AUC: ', roc_auc_score(y_test, pred))


# In[ ]:



