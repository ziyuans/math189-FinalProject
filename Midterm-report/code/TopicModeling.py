# Topic Modeling.py

# CS189 Math of Big Data project
# Name: Shihao Lin, Ziyuan Shang


import pandas as pd
import numpy as np
import json
import os
import seaborn as sns
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
from collections import defaultdict
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


if __name__ == '__main__':
#importing every dataset
parent_dir = 'Desktop/dataset/'
busi_state=pd.read_csv(parent_dir +"busi_s.csv")
busi_a=pd.read_csv(parent_dir +"business_attributes.csv")
busi_ac=pd.read_csv(parent_dir +"busi_a_cleaned.csv")
busi_new=pd.read_csv(parent_dir +"busi_new.csv")
rev =pd.read_csv(parent_dir +"review_cleaned.csv")

# merge busi_new on 'business_id'
busiReview = rev.merge(busi_new, on = 'business_id', how = 'inner')
colsb = ['Unnamed: 0_x','Unnamed: 0_y']
busiReview= busiReview.drop(colsb, axis=1)
busiReview= busiReview.dropna(subset = ['stars_y'])

busiReview['text']=busiReview['text'].apply(lambda x:str(x).replace('b',''))
busiReview['text']=busiReview['text'].apply(lambda x:str(x).replace('\'',''))

# Plot the correlation matrix
def corr_plot(df, title = 'Correlation Matrix', annot=False, show = True):
    sns.set(style = 'white')

    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, annot=annot,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title(title)
    if show:
        plt.show()
corr_plot(busiReview)

# prepare documents
documents = [(t, star) for t,star in zip(busiReview['text'], busiReview['stars_x'])]
print(documents[0])

# test
ps = PorterStemmer()
example_words = ["python","pythoner","pythoning","pythoned","pythonly"]

for w in example_words:
    print(ps.stem(w))

# get train and text set - small scale 3000
training_set = documents[:3000]
testing_set = documents[3000:]

# count words frequency 
all_words = []

for (t, star) in training_set:
    for word in t.split():
        w = word.lower().replace('.', '').replace(',', '').replace('!', '')
        all_words.append(ps.stem(w))
        
all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15))
print("stupid appeared: " + str(all_words['stupid']) + "times")


# generate word features
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

# encode Pos and neg 
temp = []
pos_count = 0
for i in range(len(featuresets)):
    if featuresets[i][1] >= 4:
        temp.append((featuresets[i][0], 'pos'))
        pos_count = pos_count + 1
    else:
        temp.append((featuresets[i][0], 'neg'))
print("pos rate: ", pos_count / len(temp))
print("len of temp: ", len(temp))


# get train and test set from the small 300 scale data
training_set  = temp[:1500]
testing_set = temp[1500:]

# run algo
Naive_Bayes_clf = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(Naive_Bayes_clf, testing_set))*100)
Naive_Bayes_clf.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MultinomialNB accuracy percent:",nltk.classify.accuracy(MNB_classifier, testing_set))

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BernoulliNB accuracy percent:",nltk.classify.accuracy(BNB_classifier, testing_set))

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

