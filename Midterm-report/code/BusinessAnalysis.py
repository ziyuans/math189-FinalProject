#Business Analysis.py

# CS189 Math of Big Data project
# Name: Shihao Lin, Ziyuan Shang
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from collections import Counter, defaultdict
import numpy as np
import chardet
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split, KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

sns.set(style='white', context='notebook', palette='deep')

# Importing dataset
data = pd.read_csv("busi_new.csv")
data.isnull().sum()

# Replace All Null Data in NaN
data = data.fillna(np.nan)
# data=data.dropna(subset = ['Price'])

# Get data types
data.dtypes

# Peek at data
data.head(10)

# Reformat Column We Are Predicting
data['Rating']=data['Rating'].map({'Negative' : 0, 'Positive': 1, 'Negative': 0, 'Positive': 1})
data.head(4)

# Identify Numeric features
numeric_features = ['review_count','Reservation','Price']

# Identify Categorical features
cat_features = ['city','Ambience','Parking','GoodForMeal']

# Count of GoodRating & <= BadRating
sns.countplot(data['Rating'],label="Count")
sns.plt.show()

# Correlation matrix between numerical values
g = sns.heatmap(data[numeric_features].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
sns.plt.show()

# Explore Review count vs Rating
g = sns.barplot(x="review_count",y="Rating",data=data)
g = g.set_ylabel("Rating positive Probability")
sns.plt.show()

# Explore Reservation vs Rating
g = sns.barplot(x="Reservation",y="Rating",data=data)
g = g.set_ylabel("Rating positive Probability")
sns.plt.show()

# Explore Price vs Rating
g = sns.barplot(x="Price",y="Rating",data=data)
g = g.set_ylabel("Rating positive Probability")
sns.plt.show()

# Explore Ambience vs Rating
g = sns.barplot(x="Ambience",y="Rating",data=data)
g = g.set_ylabel("Positive Rating Probability")
sns.plt.show()

# Explore city vs Rating
g = sns.barplot(x="city",y="Rating",data=data)
g = g.set_ylabel("Positive Rating Probability")
sns.plt.show()

# Explore Parking vs Rating
g = sns.barplot(x="Parking",y="Rating",data=data)
g = g.set_ylabel("Positive Rating Probability")
sns.plt.show()

# Explore GoodForMeal vs Rating
g = sns.barplot(x="GoodForMeal",y="Rating",data=data)
g = g.set_ylabel("Positive Rating Probability")
sns.plt.show()

####################################################
############### FEATURE ENGINEERING ################
####################################################

# Create Parking Column - Binary Yes(1) or No(0)
data["Parking"] = data["Parking"].replace(['lot','street','garage','valet','validated'], 'Available')
data["Parking"] = data["Parking"].map({"Available":1, "None":0})
data["Parking"] = data["Parking"].astype(int)
# Drop the data you don't want to use
data.drop(labels=["GoodForMeal","Unnamed: 0","business_id","city","stars","state","Ambience","GoodForMeal","Price"], axis = 1, inplace = True)
print('Dataset with Dropped Labels')
print(data.head())

###################################################
##################### MODELING #####################
####################################################
# Split-out Validation Dataset and Create Test Variables
array = data.values
X = array[:,0:3]
Y = array[:,3]
print('Split Data: X')
print(X)
print('Split Data: Y')
print(Y)
validation_size = 0.25
seed = 7
num_folds = 10
scoring = 'accuracy'
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,
    test_size=validation_size,random_state=seed)

# Params for Random Forest
num_trees = 100
max_features = 3

#Spot Check 5 Algorithms (LR, LDA, KNN, CART, GNB, SVM)
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier(n_estimators=num_trees, max_features=max_features)))
#models.append(('SVM', SVC()))
# evalutate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


fig = plt.figure()
fig.suptitle('Algorith Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()