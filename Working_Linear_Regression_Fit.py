from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics
import warnings
import joblib

warnings.simplefilter(action='ignore', category=FutureWarning)

raw_data1 = pd.read_csv('C:/Users/micha/Desktop/Linear_Regression_1.csv')
raw_data2 = pd.read_csv('C:/Users/micha/Desktop/Linear_Regression_2.csv')
raw_data3 = pd.read_csv('C:/Users/micha/Desktop/Linear_Regression_3.csv')
raw_data4 = pd.read_csv('C:/Users/micha/Desktop/Linear_Regression_4.csv')
raw_data5 = pd.read_csv('C:/Users/micha/Desktop/Linear_Regression_5.csv')
raw_data6 = pd.read_csv('C:/Users/micha/Desktop/Linear_Regression_6.csv')
raw_data7 = pd.read_csv('C:/Users/micha/Desktop/Linear_Regression_7.csv')
raw_data8 = pd.read_csv('C:/Users/micha/Desktop/Linear_Regression_8.csv')
raw_data9 = pd.read_csv('C:/Users/micha/Desktop/Linear_Regression_9.csv')
raw_data10 = pd.read_csv('C:/Users/micha/Desktop/Linear_Regression_10.csv')
raw_data11 = pd.read_csv('C:/Users/micha/Desktop/Linear_Regression_11.csv')
raw_data12 = pd.read_csv('C:/Users/micha/Desktop/Linear_Regression_12.csv')
raw_data = pd.concat([raw_data1, raw_data2, raw_data3, raw_data4, raw_data5, raw_data6, raw_data7, raw_data8, raw_data9, raw_data10, raw_data11, raw_data12])

cleaned_data = raw_data.drop(['Fighter','WeightClass', 'My Proj','Consistency','Value'], axis=1) #put cost back

# # print(cleaned_data.dtypes)

x= raw_data[['Fight #','Odds', 'Win %','Cost','Avg Score', 'ITD Odds']]
y= raw_data['Actual Score']



clf=LinearRegression()

clf.fit(x,y)


joblib.dump(clf, 'Real1.pkl')