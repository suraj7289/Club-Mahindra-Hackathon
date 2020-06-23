# -*- coding: utf-8 -*-
"""
Created on Fri May  3 23:50:41 2019

@author: surakkum
"""

import numpy as np
import os
import pandas as pd
import datetime


print(os.getcwd())
os.chdir('C:\\Users\\surakkum\\Desktop\\Python\\Dataset\\Analytics Vidhya')

train = pd.read_csv('train.csv',index_col = 'reservation_id')

train[train['roomnights']== -45] ##get index of row having -45 in roomnights
train = train.drop(['8cf7476b7111e0f969ef00d582f8a0833794239ebd32067d57451b3bdab22dc2'])

train[train['roomnights']== 60].index ##get index of row having 60 in roomnights
train = train.drop(['d6b48086b9ba5403d5103cb3587b933917cb4c87aa2c329c74c8207da47fe676'])


train.groupby(['resort_type_code'])['amount_spent_per_room_night_scaled'].mean()



x = train.iloc[:,0:-1]
y = train.iloc[:,-1]
x['checkin_month'] = pd.to_datetime(x['checkin_date']).dt.month


def EncodeAgeBucket(row):
    
    return ord(row['member_age_buckets']) -65
 
from datetime import date,datetime
from dateutil.parser import parse

def DiffOfDate(row,LowDate,HighDate):
    
    dt1 = datetime.strptime(row[LowDate], '%d/%m/%y')
    dt2 = datetime.strptime(row[HighDate], '%d/%m/%y')
    
    return (dt2 -dt1 ).total_seconds()/(86400)

def PreProcessing(X,training = True):
    X['SameState'] = 0
    X.loc[X['state_code_residence']!=X['state_code_resort'],'SameState'] = 1
    X['member_age_buckets'] = X.apply(EncodeAgeBucket,axis =1 )
    X['Stay_in_days']= X.apply(DiffOfDate,axis = 1,args =('checkin_date','checkout_date'))
    X['season_holidayed_code'].fillna(value = X['season_holidayed_code'].value_counts().index[0], inplace=True)
    X['state_code_residence'].fillna(8.0, inplace = True)
    X['Booked_before_days'] = X.apply(DiffOfDate,axis = 1,args =('booking_date','checkin_date'))
    X['Planned'] = 0
    X.loc[X['Booked_before_days']>15, 'Planned']=1
    X['Years_old'] = X.apply(Age,axis = 1)
    X=X.loc[:,('main_product_code','room_type_booked_code','total_pax','roomnights','cluster_code','season_holidayed_code','checkin_month','channel_code','persontravellingid','member_age_buckets','Years_old','state_code_residence','Stay_in_days','SameState','booking_type_code','state_code_resort','Planned','numberofchildren','resort_type_code')]
    cols_to_encode = ['main_product_code','cluster_code','season_holidayed_code','checkin_month','channel_code','persontravellingid','room_type_booked_code','state_code_residence','booking_type_code','state_code_resort']
    for i in cols_to_encode:
        X = pd.concat([X.drop(i, axis=1), pd.get_dummies(X[i], prefix = i,drop_first= True)], axis=1)

    return X

def Age(row):
    currdttm = date.today()
    bdate = dt = row['checkin_date']
    bdttm = datetime.strptime(bdate, '%d/%m/%y')
    bdttm = datetime.date(bdttm)
    
    return (currdttm -bdttm ).total_seconds()/(365*86400)

    
x_third = PreProcessing(x)
    
from xgboost import XGBRegressor
xgb_model = XGBRegressor()
xgb_model.fit(x_third, y)

test = pd.read_csv('test.csv',index_col ='reservation_id')
test['checkin_month'] = pd.to_datetime(test['checkin_date']).dt.month

X_test = PreProcessing(test)
X_test['amount_spent_per_room_night_scaled'] = xgb_model.predict(X_test)
out = pd.DataFrame(X_test['amount_spent_per_room_night_scaled'])
out.to_csv('output_xgb_3.csv')

