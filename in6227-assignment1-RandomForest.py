# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 2023
@author: mau001
Assignment 1 - Random Forest
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Numeric Constants
numeric_list = [0, 2, 4, 10, 11, 12]

# Pre-processing of Training Dataset
dataset_train = pd.read_csv('adult-full.data', header= None, na_values='?')

# Pre-processing to drop unknown fields with '?'
dataset_train = dataset_train.dropna()

dataset_train_size = len(dataset_train)
featurelen = dataset_train.shape[1]
featuresets = dataset_train
print ("[Train] Dataset (Row, Column): " + str(dataset_train.shape))

# Pre-processing to convert non-numeric values to numeric representations
label_encoders = []
for i in range(0, featurelen):
    if not i in numeric_list:
        label_encoder = LabelEncoder()
        label_encoders.append(label_encoder.fit(dataset_train[i]))
    else:
        label_encoders.append("")

for i in range(0, featurelen):
    if not i in numeric_list:
        featuresets[i] = label_encoders[i].transform(dataset_train[i])
    else:
        featuresets[i] = dataset_train.values[:,i]

train_set = featuresets

# Seperating the X and Y Values
X_train = train_set.values[:,0:featurelen-1]
y_train = train_set.values[:,featurelen-1]
y_train = y_train.astype(str)

# Pre-processing of Testing Dataset
dataset_test = pd.read_csv('adult.test', header= None, na_values='?')

# Pre-processing to drop unknown fields with '?'
dataset_test = dataset_test.dropna()
dataset_test_size = len(dataset_test)

# Pre-processing to remove '.' at the end of each test row
dataset_test[featurelen-1] = dataset_test[featurelen-1].str.strip('.')

featurelen = dataset_test.shape[1]
print ("[Test] Dataset (Row, Column): " + str(dataset_test.shape))

featuresets = dataset_test

for i in range(0, featurelen):
    if not i in numeric_list:
        featuresets[i] = label_encoders[i].transform(dataset_test[i])
    else:
        featuresets[i] = dataset_test.values[:,i]

test_set = featuresets

# Seperating the X and Y Values
X_test = test_set.values[:,0:featurelen-1]
y_test = test_set.values[:,featurelen-1]
y_test = y_test.astype(str)

##############################################################################
# # Using GridSearchCV to find what is the best combination of paramters for RandomForest
# # Define the parameter grid to search over
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [10, 20, 30, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['sqrt', 'log2']
# }

# rfc = RandomForestClassifier(random_state=42)

# # Perform grid search to find optimal hyperparameters
# grid_search = GridSearchCV(rfc, param_grid=param_grid, cv=5)
# grid_search.fit(X_train, y_train)

# print("Best hyperparameters: ", grid_search.best_params_)
# print("Best cross-validated score: ", grid_search.best_score_)

# # Results:
# # Best hyperparameters:  {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 200}
# # Best cross-validated score:  0.8650841343206613
##############################################################################


# Create and train RandomForestClassfier
rfc = RandomForestClassifier(n_estimators=200, max_depth=20, max_features='sqrt', min_samples_leaf=2, 
                             min_samples_split=10)

rfc.fit(X_train, y_train)
result = rfc.predict(X_test)

# Printing results
ctr = 0
ctr_true = 0
ctr_false = 0
for i in range(0,len(result)):
    if y_test[i] == result[i]:
        ctr = ctr + 1
    if result[i] == '0':
        ctr_true = ctr_true + 1
    if result[i] == '1':
        ctr_false = ctr_false + 1
        
print ("[Result] Score: " + str(rfc.score(X_test,y_test)))
print ("[Result] " + str(ctr) + " of " + str(len(result)) + " (<= 50k: " + str(ctr_true) + ", >50k: " + str(ctr_false) + ")")

# calculate precision and recall scores
accuracy = accuracy_score(y_test, result)
precision = precision_score(y_test, result, average='weighted')
recall = recall_score(y_test, result, average='weighted')

print("[Result] Accuracy: " + str(accuracy))
print("[Result] Precision: " + str(precision))
print("[Result] Recall: " + str(recall))