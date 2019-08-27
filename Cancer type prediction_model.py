# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:00:24 2019
Topic : Type of breast Cancer ( benign or malignant)
Data Source: Kaggle
@author: KRaj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data 
data=pd.read_csv(r"D:\EDA\Breast_Cancer_Kaggle.csv")

#Wrangling of data
data.head(10)
data.describe()
data.shape #(569,32)

# checking the counts of categorical feature
data.describe(include=object) # we have only one feature as categorical
# inspect individual colums
data['diagnosis'].value_counts() # there is no null value
data=data.drop(data.columns[0], axis=1)
# looking for null values in all the columns
data.isnull().sum() # no null value found

# Convert(lable the categorical variable 
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
data['diagnosis']=encoder.fit_transform(data['diagnosis'])

# checking the co-relation
plt.figure(figsize=(20,12))
sns.heatmap(data.corr(),annot=True,fmt='.0%')

#spliting the data in dependent and independent variables

y= data.iloc[:,0].values
x=data.drop(data.columns[0], axis=1)

## Scaling the feature 

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_x= scaler.fit_transform(x)

# spliting the data 

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25, random_state=0)

# Model selection


# Desicion Tree
# Support vectoor Machine (SVM)

# Logistic regression
from sklearn.linear_model import LogisticRegression

classifier= LogisticRegression()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

# validating the model acuracy

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

accuracy_score(y_test,y_pred)
# 95 %
classification_report(y_test,y_pred)
# F1 score - 90 %

# Random forest
from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(n_estimators=10, max_depth = 3)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

accuracy_score(y_test,y_pred)
# 97 %
classification_report(y_test,y_pred)
# F1 score - 90 %