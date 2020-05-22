# -*- coding: utf-8 -*-
import pandas as pd
import warnings 
from sklearn.neighbors import KNeighborsClassifier
from geopy.distance import vincenty
from datetime import date
warnings.filterwarnings("ignore")

def monthsBeforeHoliday(dataset, holiday):
    holidayTokens= holiday.split('-')
    holidaydate=date(2016,int(holidayTokens[0]),int(holidayTokens[1]))
    weeksbefore=[]
    for i in range(0,dataset.shape[0]):
        dateTokens=dataset[i,0].split('-')
        departuredate=date(2016,int(dateTokens[1]),int(dateTokens[2]))
        diffDays=(holidaydate-departuredate)
        diffDays=round(diffDays.days/7.0,0)
        weeksbefore.append(diffDays)
    return weeksbefore

def weeksBeforeHoliday(dataset, holiday):
    holidayTokens= holiday.split('-')
    holidaydate=date(2016,int(holidayTokens[0]),int(holidayTokens[1]))
    daysbefore=[]
    for i in range(0,dataset.shape[0]):
        dateTokens=dataset[i,0].split('-')
        departuredate=date(2016,int(dateTokens[1]),int(dateTokens[2]))
        diffDays=(holidaydate-departuredate)
        diffDays=round(diffDays.days/30.41,0)
        daysbefore.append(diffDays)
    return daysbefore

def weekday(dataset):
    weekday=[]
    for i in range(0,dataset.shape[0]):
         dateTokens=dataset[i,0].split('-')
         onedate=date(int(dateTokens[0]),int(dateTokens[1]),int(dateTokens[2]))
         weekday.append(onedate.weekday())
    
    return weekday

#load datasets
df_train =pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
#keeping weekstod and std_wtd and then one hot encoding dep and arr
y_train = df_train[['PAX']]
df_train_co= df_train.as_matrix(['LongitudeDeparture','LatitudeDeparture','LongitudeArrival','LatitudeArrival'])
df_train_dates= df_train.as_matrix(['DateOfDeparture'])

df_test_co= df_test.as_matrix(['LongitudeDeparture','LatitudeDeparture','LongitudeArrival','LatitudeArrival'])
df_test_dates= df_test.as_matrix(['DateOfDeparture'])
# find vincenty distance to create new feature distance on train
dist_train=[]
for i in range(0,8899):
    dep_air_train= (df_train_co[i,0],df_train_co[i,1])
    arr_air_train= (df_train_co[i,2],df_train_co[i,3])
    dist_train.append(vincenty(dep_air_train,arr_air_train).km)
    
df_dist_train= pd.DataFrame({'Distance': dist_train})
# find vincenty distance to create new feature distance on test
dist_test=[]
for i in range(0,2229):
    dep_air_test= (df_test_co[i,0],df_test_co[i,1])
    arr_air_test= (df_test_co[i,2],df_test_co[i,3])
    dist_test.append(vincenty(dep_air_test,arr_air_test).km)

df_dist_test= pd.DataFrame({'Distance': dist_test})

df_train.drop(df_train.columns[[0,2,3,4,6,7,8,11]], axis=1, inplace=True)
df_train= pd.concat([df_train,pd.get_dummies(df_train['Departure'],prefix='Departure'),pd.get_dummies(df_train['Arrival'],prefix='Arrival')],axis=1)
df_train.drop(['Departure'],axis=1,inplace=True)
df_train.drop(['Arrival'],axis=1,inplace=True)

df_train_w_dist= pd.concat([df_train,df_dist_train], axis=1)

df_test.drop(df_test.columns[[0,2,3,4,6,7,8]], axis=1, inplace=True)
df_test= pd.concat([df_test,pd.get_dummies(df_test['Departure'],prefix='Departure'),pd.get_dummies(df_test['Arrival'],prefix='Arrival')],axis=1)
df_test.drop(['Departure'],axis=1,inplace=True)
df_test.drop(['Arrival'],axis=1,inplace=True)

df_test_w_dist= pd.concat([df_test,df_dist_test],axis=1)
#special days
tygiving='10-25' #thangsgiving
mday='05-11' #mother's day
iday='07-04' #independence day
fday='06-18' #father's day
hween='10-31' #halloween
vday='02-14' #valentine's day
nye='12-31' #new year's eve
xmas='12-25'#Xmas

#TRAIN MONTHS
#Valentine's Day
df_train_months_b_vday= monthsBeforeHoliday(df_train_dates,vday)
#Mother's Day
df_train_months_b_mday= monthsBeforeHoliday(df_train_dates,mday)
#Father's Day
df_train_months_b_fday= monthsBeforeHoliday(df_train_dates,fday)
#Independence Day
df_train_months_b_iday= monthsBeforeHoliday(df_train_dates,iday)
#Thangsgiving
df_train_months_b_thg= monthsBeforeHoliday(df_train_dates,tygiving)
#Halloween
df_train_months_b_hal= monthsBeforeHoliday(df_train_dates,hween)
#Xmas
df_train_months_b_xmas= monthsBeforeHoliday(df_train_dates,xmas)
#New Year's Eve
df_train_months_b_nye= monthsBeforeHoliday(df_train_dates,nye)

#TEST MONTHS
#Valentine's Day
df_test_months_b_vday= monthsBeforeHoliday(df_test_dates,vday)
#Mother's Day
df_test_months_b_mday= monthsBeforeHoliday(df_test_dates,mday)
#Father's Day
df_test_months_b_fday= monthsBeforeHoliday(df_test_dates,fday)
#Independence Day
df_test_months_b_iday= monthsBeforeHoliday(df_test_dates,iday)
#Thangsgiving
df_test_months_b_thg= monthsBeforeHoliday(df_test_dates,tygiving)
#Halloween
df_test_months_b_hal= monthsBeforeHoliday(df_test_dates,hween)
#Xmas
df_test_months_b_xmas= monthsBeforeHoliday(df_test_dates,xmas)
#New Year's Eve
df_test_months_b_nye= monthsBeforeHoliday(df_test_dates,nye)

#TRAIN WEEKS
#Valentine's Day
df_train_weeks_b_vday= weeksBeforeHoliday(df_train_dates,vday)
#Mother's Day
df_train_weeks_b_mday= weeksBeforeHoliday(df_train_dates,mday)
#Father's Day
df_train_weeks_b_fday= weeksBeforeHoliday(df_train_dates,fday)
#Independence Day
df_train_weeks_b_iday= weeksBeforeHoliday(df_train_dates,iday)
#Thangsgiving
df_train_weeks_b_thg= weeksBeforeHoliday(df_train_dates,tygiving)
#Halloween
df_train_weeks_b_hal= weeksBeforeHoliday(df_train_dates,hween)
#Xmas
df_train_weeks_b_xmas= weeksBeforeHoliday(df_train_dates,xmas)
#New Year's Eve
df_train_weeks_b_nye= weeksBeforeHoliday(df_train_dates,nye)

#TEST WEEKS
#Valentine's Day
df_test_weeks_b_vday= weeksBeforeHoliday(df_test_dates,vday)
#Mother's Day
df_test_weeks_b_mday= weeksBeforeHoliday(df_test_dates,mday)
#Father's Day
df_test_weeks_b_fday= weeksBeforeHoliday(df_test_dates,fday)
#Independence Day
df_test_weeks_b_iday= weeksBeforeHoliday(df_test_dates,iday)
#Thangsgiving
df_test_weeks_b_thg= weeksBeforeHoliday(df_test_dates,tygiving)
#Halloween
df_test_weeks_b_hal= weeksBeforeHoliday(df_test_dates,hween)
#Xmas
df_test_weeks_b_xmas= weeksBeforeHoliday(df_test_dates,xmas)
#New Year's Eve
df_test_weeks_b_nye= weeksBeforeHoliday(df_test_dates,nye)
#convert list to pd dataframe
df_vday_train= pd.DataFrame({'monthsBeforeValentines': df_train_months_b_vday})
df_mday_train= pd.DataFrame({'monthsBeforeMothers': df_train_months_b_mday})
df_fday_train= pd.DataFrame({'monthsBeforeFathers': df_train_months_b_fday})
df_iday_train= pd.DataFrame({'monthsBeforeIndependence': df_train_months_b_iday})
df_thg_train= pd.DataFrame({'monthsBeforeThanksgiving': df_train_months_b_thg})
df_hal_train= pd.DataFrame({'monthsBeforeHalloween': df_train_months_b_hal})
df_xmas_train= pd.DataFrame({'monthsBeforeXmas': df_train_months_b_xmas})
df_nye_train= pd.DataFrame({'monthsBeforeNYE': df_train_months_b_nye})

df_hol_train= pd.concat([df_vday_train,df_mday_train,df_fday_train,df_iday_train,df_thg_train,df_hal_train,df_xmas_train,df_nye_train], axis=1)

df_vday_test= pd.DataFrame({'monthsBeforeValentines': df_test_months_b_vday})
df_mday_test= pd.DataFrame({'monthsBeforeMothers': df_test_months_b_mday})
df_fday_test= pd.DataFrame({'monthsBeforeFathers': df_test_months_b_fday})
df_iday_test= pd.DataFrame({'monthsBeforeIndependence': df_test_months_b_iday})
df_thg_test= pd.DataFrame({'monthsBeforeThanksgiving': df_test_months_b_thg})
df_hal_test= pd.DataFrame({'monthsBeforeHalloween': df_test_months_b_hal})
df_xmas_test= pd.DataFrame({'monthsBeforeXmas': df_test_months_b_xmas})
df_nye_test= pd.DataFrame({'monthsBeforeNYE': df_test_months_b_nye})


df_hol_test= pd.concat([df_vday_test,df_mday_test,df_fday_test,df_iday_test,df_thg_test,df_hal_test,df_xmas_test,df_nye_test], axis=1)

df_vday_train= pd.DataFrame({'weeksBeforeValentines': df_train_weeks_b_vday})
df_mday_train= pd.DataFrame({'weeksBeforeMothers': df_train_weeks_b_mday})
df_fday_train= pd.DataFrame({'weeksBeforeFathers': df_train_weeks_b_fday})
df_iday_train= pd.DataFrame({'weeksBeforeIndependence': df_train_weeks_b_iday})
df_thg_train= pd.DataFrame({'weeksBeforeThanksgiving': df_train_weeks_b_thg})
df_hal_train= pd.DataFrame({'weeksBeforeHalloween': df_train_weeks_b_hal})
df_xmas_train= pd.DataFrame({'weeksBeforeXmas': df_train_weeks_b_xmas})
df_nye_train= pd.DataFrame({'weeksBeforeNYE': df_train_weeks_b_nye})

df_hol_train= pd.concat([df_hol_train,df_vday_train,df_mday_train,df_fday_train,df_iday_train,df_thg_train,df_hal_train,df_xmas_train,df_nye_train], axis=1)

df_vday_test= pd.DataFrame({'weeksBeforeValentines': df_test_weeks_b_vday})
df_mday_test= pd.DataFrame({'weeksBeforeMothers': df_test_weeks_b_mday})
df_fday_test= pd.DataFrame({'weeksBeforeFathers': df_test_weeks_b_fday})
df_iday_test= pd.DataFrame({'weeksBeforeIndependence': df_test_weeks_b_iday})
df_thg_test= pd.DataFrame({'weeksBeforeThanksgiving': df_test_weeks_b_thg})
df_hal_test= pd.DataFrame({'weeksBeforeHalloween': df_test_weeks_b_hal})
df_xmas_test= pd.DataFrame({'weeksBeforeXmas': df_test_weeks_b_xmas})
df_nye_test= pd.DataFrame({'weeksBeforeNYE': df_test_weeks_b_nye})

df_hol_test= pd.concat([df_hol_test,df_vday_test,df_mday_test,df_fday_test,df_iday_test,df_thg_test,df_hal_test,df_xmas_test,df_nye_test], axis=1)

df_numday_train= weekday(df_train_dates)
df_numday_test= weekday(df_test_dates)

df_weekday_train = pd.DataFrame({'Weekday': df_numday_train})
df_weekday_test = pd.DataFrame({'Weekday': df_numday_test})

df_fe_train= pd.concat([df_hol_train,df_weekday_train], axis=1)
df_fe_test= pd.concat([df_hol_test,df_weekday_test], axis=1)

df_fin_train= pd.concat([df_train_w_dist,df_fe_train],axis=1)
df_fin_test= pd.concat([df_test_w_dist,df_fe_test],axis=1)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(df_fin_train, y_train)
y_pred = knn.predict(df_fin_test)
#writing csv
y_pred_df = pd.DataFrame(y_pred)
y_pred_df.index.name = 'Id'
y_pred_df.columns = ['Label']
y_pred_df.to_csv('knn_pred.csv')