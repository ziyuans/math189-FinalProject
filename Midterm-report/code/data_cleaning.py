# Data_cleanning.py

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

def read_files():
    d = defaultdict(list)

    parent_dir = 'Desktop/dataset/'

    path_dic = {'B': 'business.csv', 'R':'review.csv','U':'user.csv'}

    for key in path_dic:
        d[key] = pd.read_csv(parent_dir + path_dic[key])
    return d

if __name__ == '__main__':
    d = read_files()

    # Clean Review.csv 
    columns = ['date', 'funny', 'useful']
    d['R'] = d['R'].drop('columns', axis=1)
    outputFile=os.path.join('Desktop/dataset/review_cleaned.csv')
    d['R'].to_csv(outputFile)

    # Clean User.csv 
    cols = ['friends','compliment_plain','compliment_more', 'compliment_writer',\
            'compliment_funny', 'compliment_profile','yelping_since','elite',\
        'compliment_list', 'cool','fans','funny','compliment_cool','compliment_cute','name']
    d['U'] = d['U'].drop(cols, axis=1)
    outputFile=os.path.join('Desktop/dataset/users_cleaned.csv')
    d['U'].to_csv(outputFile)

    # Clean Business.csv 
    colsu = ['hours.Sunday','hours.Monday', 'hours.Tuesday','hours.Wednesday', \
            'hours.Thursday','hours.Friday','hours.Saturday', 'address',\
            'attributes.HairSpecializesIn.extensions',\
            'attributes.BusinessAcceptsBitcoin','attributes.RestaurantsCounterService', \
            'attributes.HairSpecializesIn.kids','attributes.HairSpecializesIn.coloring', \
            'attributes.AcceptsInsurance','postal_code', 'attributes.HairSpecializesIn.perms',\
            'attributes.HairSpecializesIn.asian','attributes.HairSpecializesIn.curly', \
            'attributes.HairSpecializesIn.straightperms',\
            'attributes.HairSpecializesIn.africanamerican','attributes.BestNights.wednesday', \
            'attributes.BestNights.sunday','attributes.BestNights.monday',\
            'attributes.BestNights.tuesday','attributes.BestNights.thursday',\
            'attributes.BestNights.friday','attributes.BestNights.saturday',
            'attributes.Corkage','attributes.Music.background_music',\
            'attributes.BYOBCorkage', 'attributes.CoatCheck',\
            'attributes.Music.background_music','attributes.Music.dj',\
            'attributes.Music.karaoke','attributes.GoodForDancing','attributes.Music.video',\
            'attributes.Music.no_music','attributes.BYOB','attributes.HasTV',\
            'attributes.Music.live','attributes.Music.jukebox','attributes.WiFi', \
            'attributes.WheelchairAccessible','attributes.Open24Hours',\
            'attributes.DogsAllowed', 'attributes.NoiseLevel','attributes.RestaurantsTakeOut',\
            'attributes.Alcohol','attributes.HappyHour','attributes.DriveThru',\
            'attributes.BikeParking','attributes.BusinessAcceptsCreditCards','attributes.Caters',\
            'attributes.GoodForKids','attributes.ByAppointmentOnly','attributes.OutdoorSeating',\
            'attributes.RestaurantsAttire','attributes.RestaurantsDelivery',\
            'attributes.RestaurantsGoodForGroups','attributes.RestaurantsTableService','attributes.Smoking']
    d['B'] = d['B'].drop(colsu, axis=1)
    # change all NaN value into None type 
    d['B'] = d['B'].astype(object).where(pd.notnull(d['B']),None)
    # sort columns
    d['B'] = d['B'][sorted(d['B'].columns)]
    # filter business that is not open 
    df_open = d['B'][d['B']['is_open'] == 1]

    #split business.csv into attributes and other information
    df_a = df_open.iloc[:, 1:48]
    outputFile=os.path.join('Desktop/dataset/business_attributes.csv')
    df_a.to_csv(outputFile)

    df_b = df_open.iloc[:, 48:]
    colsdb = ['is_open','name','neighborhood']
    df_b = df_b.drop(colsdb, axis=1)
    outputFile=os.path.join('Desktop/dataset/business_cleaned.csv')
    df_b.to_csv(outputFile)

