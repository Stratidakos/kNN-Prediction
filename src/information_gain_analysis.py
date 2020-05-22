# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np                     # For mathematical calculations
import seaborn as sns                  # For data visualization
import matplotlib.pyplot as plt        # For plotting graphs
import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore")

#load datasets
df_train =pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
#copy datasets
train_original = df_train.copy()
test_original = df_test.copy()
#understanding data
print(df_train.keys())
print(df_train.dtypes)
#visualisation of classes(frequency table, percentage distribution
# and bar plot.)
print(df_train['PAX'].value_counts())
print(df_train['PAX'].value_counts(normalize=True))
print(df_train['PAX'].value_counts().plot.bar())
#visualisation of categorial features
plt.figure(2)
#plt.subplot(221)
#df_train['DateOfDeparture'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='DateOfDeparture')

plt.subplot(221)
df_train['Departure'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='Departure')

#plt.subplot(223)
#df_train['CityDeparture'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='CityDeparture')

plt.subplot(222)
df_train['Arrival'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='Arrival')

#plt.subplot(225)
#df_train['CityArrival'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='CityArrival')

#visualisation of numerical features
plt.figure(3)
plt.subplot(321)
sns.distplot(df_train['std_wtd'])

plt.subplot(322)
sns.distplot(df_train['WeeksToDeparture'])

plt.figure(4)
plt.subplot(441)
sns.distplot(df_train['LongitudeDeparture'])

plt.subplot(442)
sns.distplot(df_train['LatitudeDeparture'])

plt.subplot(443)
sns.distplot(df_train['LongitudeArrival'])

plt.subplot(444)
sns.distplot(df_train['LatitudeArrival'])

#check for missing values
print(df_train.info())
print(df_train.isnull().sum())

#searching for correlation between features
matrix = df_train.corr()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(matrix, vmax=1.0, square=True, cmap="BuPu");