# EDA.py

# CS189 Math of Big Data project
# Name: Shihao Lin, Ziyuan Shang

import numpy as np
import pandas as pd
import gc
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib.gridspec as gridspec 
import matplotlib.gridspec as gridspec 
import networkx as nx
import heapq  # for getting top n number of things from list,dict
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn import preprocessing
from collections import defaultdict

if __name__ == '__main__':
#settings
start_time=time.time()
color = sns.color_palette()
sns.set_style("white")
warnings.filterwarnings("ignore")

#importing every cleaned dataset
parent_dir = 'Desktop/dataset/'
busi=pd.read_csv(parent_dir +"business_cleaned.csv")
busi_a=pd.read_csv(parent_dir +"business_attributes.csv")
rev =pd.read_csv(parent_dir +"review_cleaned.csv")
user=pd.read_csv(parent_dir +"users_cleaned.csv")

# Get an general sense of the data 
## How many users make reviews?
print('Total users:', user['user_id'].nunique())
## How many user_id's contains in reviews
print('Total users review:', rev['user_id'].nunique())
print('Total organization:', busi['business_id'].nunique())
## How many organizations has reviews?
print('Total organizations with review:', rev['business_id'].nunique())
rev['business_id'].nunique()/busi['business_id'].nunique())
negative_reviws = len(rev[rev["stars"]<4])
positive_reviews =len(rev[rev["stars"]>3])
total_reviews = len(rev)
print("Total reviews: {}".format(total_reviews))
print("Total negative(1-3) reviews: {}".format(negative_reviws))
print("Total positive(4-5) reviews: {}".format(positive_reviews))

# plot data & cleanning accordingly

# Number of Business vs States
busi['state']=busi['state'].apply(lambda x:str(x).replace('b',''))
busi['state']=busi['state'].apply(lambda x:str(x).replace('\'',''))
sns.countplot(data = busi_state, x = 'state')
plt.title("Which state has the most reviews?")
locs, labels = plt.xticks()
plt.ylabel('# of reviews', fontsize=12)
plt.xlabel('States', fontsize=12)
plt.show()
plt.close()

# Choose the five states that have the most reviews
busi_state = busi[busi['state'].isin(['AZ', 'NV', 'ON', 'NC','OH'])]
busi_state.to_csv(parent_dir+'business_s.csv')


# Distribution of the ratings
x=busi['stars'].value_counts()
x=x.sort_index()
plt.figure(figsize=(8,4))
ax= sns.barplot(x.index, x.values, alpha=0.8)
plt.title("Star Rating Distribution")
plt.ylabel('# of businesses', fontsize=12)
plt.xlabel('Star Ratings ', fontsize=12)
#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
plt.show()
plt.close()

# Number of Business vs Cities
busi_state['city']=busi_state['city'].apply(lambda x:str(x).replace('\'',''))
busi_state['city']=busi_state['city'].apply(lambda x:str(x).replace('b',''))

x=busi_state['city'].value_counts()
x=x.sort_values(ascending=False)
x=x.iloc[0:20]
plt.figure(figsize=(16,4))
ax = sns.barplot(x.index, x.values, alpha=0.8,color=color[3])
plt.title("Which city has the most reviews?")
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.ylabel('# businesses', fontsize=12)
plt.xlabel('City', fontsize=12)
#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
plt.show()
plt.close()

# group reviews by user_id 
user_agg=reviews.groupby('user_id').agg({'review_id':['count'],'stars':['mean']})
user_agg=user_agg.sort_values([('review_id','count')],ascending=False)
user_agg.head(10)

# Cap max reviews to 30 for better visuals
user_agg['review_id'].loc[user_agg['review_id']>30] = 30
plt.figure(figsize=(12,5))
gridspec.GridSpec(1,2)
plt.subplot2grid((1,2),(0,0))
# Cumulative Distribution
ax=sns.kdeplot(user_agg['review_id'],shade=True,color='r')
plt.title("How many reviews does an average user give?",fontsize=15)
plt.xlabel('# of reviews given', fontsize=12)
plt.ylabel('# of users', fontsize=12)
plt.show()
plt.close()

# generate popularity on latitud and longitude of the city  

#get all ratings data
rating_data=busi_state[['latitude','longitude','stars','review_count']]
# Creating a custom column popularity using stars*no_of_reviews
rating_data['popularity']=rating_data['stars']*rating_data['review_count']

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))

#a random point inside vegas
lat = 36.207430
lon = -115.268460
#some adjustments to get the right pic
lon_min, lon_max = lon-0.3,lon+0.5
lat_min, lat_max = lat-0.4,lat+0.5
#subset for vegas
ratings_data_vegas=rating_data[(rating_data["longitude"]>lon_min) & 
                   (rating_data["longitude"]<lon_max) & 
                   (rating_data["latitude"]>lat_min) & 
                   (rating_data["latitude"]<lat_max)]
#Facet scatter plot
ratings_data_vegas.plot(kind='scatter', x='longitude', y='latitude',
                color='red', 
                s=.02, alpha=.6, subplots=True, ax=ax1)
ax1.set_title("Las Vegas")

#Do the same thing for pheonix
lat = 33.435463
lon = -112.006989
lon_min, lon_max = lon-0.3,lon+0.5
lat_min, lat_max = lat-0.4,lat+0.5
ratings_data_pheonix=rating_data[(rating_data["longitude"]>lon_min) & 
                                 (rating_data["longitude"]<lon_max) & 
                                 (rating_data["latitude"]>lat_min) & 
                                 (rating_data["latitude"]<lat_max)]
ratings_data_pheonix.plot(kind='scatter', x='longitude', y='latitude',
                color='yellow', 
                s=.02, alpha=.6, subplots=True, ax=ax2)
ax2.set_title("Pheonix")
ax2.set_facecolor('white')
f.show()
f.close()


# Preprocess review and business.csv 
# Create new column Rating that 4-5 is positive, 1-3 is negative
rev['Rating']=np.where(rev['stars'] >=4, 'Positive', 'Negative')
busi_state['Rating']=np.where(busi_state['stars'] >=4, 'Positive', 'Negative')


#create new column Ambience according to attributes.ambience
busi_a['Ambience']=np.where(busi_a['attributes.Ambience.casual'], 'casual', 
                            np.where(busi_a['attributes.Ambience.classy'],'classy',
                            np.where(busi_a['attributes.Ambience.divey'],'divey',
                            np.where(busi_a['attributes.Ambience.hipster'],'hipster',
                            np.where(busi_a['attributes.Ambience.intimate'],'intimate',
                            np.where(busi_a['attributes.Ambience.romantic'],'romantic',
                            np.where(busi_a['attributes.Ambience.touristy'],'touristy',
                            np.where(busi_a['attributes.Ambience.trendy'],'trendy',
                            np.where(busi_a['attributes.Ambience.upscale'],'upscale','None')
                            ))))))))

#create new column Parking according to attributes.BusinessParking
busi_a['Parking']=np.where(busi_a['attributes.BusinessParking.lot'], 'lot', 
                            np.where(busi_a['attributes.BusinessParking.valet'],'valet',
                            np.where(busi_a['attributes.BusinessParking.validated'],'validated',
                            np.where(busi_a['attributes.BusinessParking.street'],'street',
                            np.where(busi_a['attributes.BusinessParking.garage'],'garage','None'

#create new column Dietary according to attributes.DietaryRestrictions
busi_a['Dietary']=np.where(busi_a['attributes.DietaryRestrictions.dairy-free'], 'dairy-free', 
                            np.where(busi_a['attributes.DietaryRestrictions.gluten-free'],'gluten-free',
                            np.where(busi_a['attributes.DietaryRestrictions.halal'],'halal',
                            np.where(busi_a['attributes.DietaryRestrictions.kosher'],'kosher',
                            np.where(busi_a['attributes.DietaryRestrictions.soy-free'],'soy-free',
                            np.where(busi_a['attributes.DietaryRestrictions.vegan'],'vegan',
                            np.where(busi_a['attributes.DietaryRestrictions.vegetarian'],'vegetarian',
                                     'None')))))))

#create new column GoodForMeal according to attributes.GoodForMeal
busi_a['GoodForMeal']=np.where(busi_a['attributes.GoodForMeal.breakfast'], 'breakfast', 
                            np.where(busi_a['attributes.GoodForMeal.dessert'],'dessert',
                            np.where(busi_a['attributes.GoodForMeal.latenight'],'latenight',
                            np.where(busi_a['attributes.GoodForMeal.brunch'],'brunch',
                            np.where(busi_a['attributes.GoodForMeal.dinner'],'dinner',
                            np.where(busi_a['attributes.GoodForMeal.lunch'],'lunch',
                                     'None'))))))

#create new column Price according to attributes.RestaurantsPriceRange2
busi_a['Price']=np.where(busi_a['attributes.RestaurantsPriceRange2']<1,'0',
                         busi_a['attributes.RestaurantsPriceRange2'])

#create new column Reservation according to attributes.RestaurantsReservations
busi_ac['Reservation']=np.where(busi_a['attributes.RestaurantsReservations'],'1','0')

# drop all used columns 
colsb = ['attributes.Ambience.casual', 'attributes.Ambience.classy',
       'attributes.Ambience.divey', 'attributes.Ambience.hipster',
       'attributes.Ambience.intimate', 'attributes.Ambience.romantic',
       'attributes.Ambience.touristy', 'attributes.Ambience.trendy',
       'attributes.Ambience.upscale', 'attributes.BusinessParking.garage',
       'attributes.BusinessParking.lot', 'attributes.BusinessParking.street',
       'attributes.BusinessParking.valet',
       'attributes.BusinessParking.validated',
       'attributes.DietaryRestrictions.dairy-free',
       'attributes.DietaryRestrictions.gluten-free',
       'attributes.DietaryRestrictions.halal',
       'attributes.DietaryRestrictions.kosher',
       'attributes.DietaryRestrictions.soy-free',
       'attributes.DietaryRestrictions.vegan',
       'attributes.DietaryRestrictions.vegetarian',
       'attributes.GoodForMeal.breakfast', 'attributes.GoodForMeal.brunch',
       'attributes.GoodForMeal.dessert', 'attributes.GoodForMeal.dinner',
       'attributes.GoodForMeal.latenight', 'attributes.GoodForMeal.lunch',
       'attributes.RestaurantsPriceRange2','attributes.RestaurantsReservations','Dietary']
busi_ac= busi_a.drop(colsb, axis=1)

# concat cleaned business data with cleaned attribues
busi_new = pd.concat([busi_state, busi_ac], axis=1)
busi_new = busi_state.merge(busi_ac, on = 'business_id', how = 'inner')

#cleaned busi_new.csv columns not needed for Business analysis
colsb = ['Unnamed: 0','Unnamed: 0.1','Counting','categories','latitude',\
         'stars','longitude','city','state']
busi_new= busi_new.drop(colsb, axis=1)
outputFile=os.path.join('Desktop/dataset/busi_new.csv')
busi_new.to_csv(outputFile)

#cleaned up none/extra data
busi_new = busi_new.astype(object).where(pd.notnull(busi_new),None)
busi['state']=busi['state'].apply(lambda x:str(x).replace('b',''))
busi_newb['Rating']=busi_newb['Rating'].apply(lambda x:str(x).replace('None','noData'))
busi_newb_dropid=busi_newb.drop('business_id', axis=1)
busi_newb_dropid['Price']=busi_newb_dropid['Price'].apply(lambda x:str(x).replace('None','0'))
busi_newb_dropid['review_count']=busi_newb_dropid['review_count'].apply(lambda x:str(x).replace('None','0'))
busi_newb_dropid=busi_newb_dropid.dropna(subset = ['stars'])

# Encode the columns into number ready for evaluation
le = preprocessing.LabelEncoder()
le.fit(busi_newb_dropid['GoodForMeal'])
busi_newb_encoded = busi_newb_dropid.apply(le.fit_transform)



