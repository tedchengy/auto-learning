#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 17:57:54 2019

@author: chen
"""
import pandas as pd
import autosklearn.classification
from sklearn.model_selection import train_test_split
from sklearn import metrics

MINUTE = 5
MEMORY = 8



data = pd.read_csv("~/data/creditcard.csv")
targets = pd.DataFrame()
targets = data['Class']
data.drop('Class', axis=1, inplace=True)

'''
data = pd.read_csv("titanic.train.csv")
targets = data['survived']
data['Gender']=data['sex'].map({'female':0,'male':1}).astype(int)
data.drop(['survived','name','ticket','embarked','cabin','sex'],axis=1, inplace=True)
'''
'''
data = pd.read_csv("500wX50_classification_data_output.csv")
targets = data['Y']
data.drop(['Y'],axis=1, inplace=True)
'''
X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.30, random_state=1)
cls = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60 * MINUTE,
                                                       ml_memory_limit=1024 * MEMORY, per_run_time_limit=60*1)
cls.fit(X_train, y_train)
predictions = cls.predict(X_test, )
print("Accuracy score", metrics.accuracy_score(y_test, predictions))
print("roc_auc_score:\n ", metrics.roc_auc_score(y_test, predictions))