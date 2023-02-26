# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 2023
@author: mau001
Assignment 1 - Decision Tree
"""

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Numeric Constants
numeric_list = [0, 2, 4, 10, 11, 12]

# Pre-processing of Training Dataset
dataset_train = pd.read_csv('adult.data', header= None, na_values='?')

# Pre-processing to drop unknown fields with '?'
dataset_train = dataset_train.dropna()

dataset_train_size = len(dataset_train)
featurelen = dataset_train.shape[1]
featuresets = dataset_train
print ("[Training] Dataset (Row, Column): " + str(dataset_train.shape))

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

# Pre-processing of Validation Dataset
dataset_validate = pd.read_csv('adult.validate', header= None, na_values='?')

# Pre-processing to drop unknown fields with '?'
dataset_validate = dataset_validate.dropna()
dataset_validate_size = len(dataset_validate)

featurelen = dataset_validate.shape[1]
print ("[Validate] Dataset (Row, Column) " + str(dataset_validate.shape))

featuresets = dataset_validate

for i in range(0, featurelen):
    if not i in numeric_list:
        featuresets[i] = label_encoders[i].transform(dataset_validate[i])
    else:
        featuresets[i] = dataset_validate.values[:,i]

validate_set = featuresets
print ("[Validate] Validate set of " + str(len(validate_set)))

# Separate labels from dataset
X_validate = validate_set.values[:,0:featurelen-1]
y_validate = validate_set.values[:,featurelen-1]
y_validate = y_validate.astype(str)


# Pre-processing of Testing Dataset
dataset_test = pd.read_csv('adult.test', header= None, na_values='?')

# Pre-processing to drop unknown fields with '?'
dataset_test = dataset_test.dropna()
dataset_test_size = len(dataset_test)

# Pre-processing to remove '.' at the end of each test row
dataset_test[featurelen-1] = dataset_test[featurelen-1].str.strip('.')

featurelen = dataset_test.shape[1] # number of columns
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
# # Using GridSearchCV to find what is the best combination of paramters for DecisionTree
# # Define the parameter grid to search over
# param_grid = {
#     'criterion': ['gini', 'entropy'],
#     'max_depth': [5, 10, 15, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt', 'log2', None]
# }

# dtc = tree.DecisionTreeClassifier(random_state=42)

# # Perform grid search to find optimal hyperparameters
# grid_search = GridSearchCV(dtc, param_grid=param_grid, cv=5, n_jobs=-1)
# grid_search.fit(X_train, y_train)

# print(f"Best parameters: {grid_search.best_params_}")
# print(f"Training accuracy: {grid_search.best_score_}")
# print(f"Test accuracy: {grid_search.score(X_test, y_test)}")

# # Results:
# # Best parameters: {'criterion': 'gini', 'max_depth': 10, 'max_features': None, 'min_samples_leaf': 2, 'min_samples_split': 2}
# # Training accuracy: 0.8535364962760174
# # Test accuracy: 0.8552914440144954
##############################################################################

# Create and train DecisionTree
dc = tree.DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=10, splitter='best', max_features=None, min_samples_leaf=2,
                               min_samples_split=2)

dc.fit(X_train, y_train)

#Perform Validation
scores = cross_val_score(dc, X_validate, y_validate,cv=10)
print ("[Validate] Cross Validate of DT-gini (score): " + str(scores))

result = dc.predict(X_test)

##############################################################################
# # Attempt to add weightage to a particular column

# # Scaler to scale input values
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)

# # Set weights for the highly correlated parameter
# weights = np.ones(X_train.shape[1])
# weights[featurelen-5] = 100

# # Train the classifier using the highly correlated parameter
# X_corr_train = (X_train[:, featurelen-5] > 7000).astype(int)[:, np.newaxis]
# X_weighted_train = X_train_scaled * weights
# X_weighted_corr_train = np.concatenate((X_corr_train, X_weighted_train), axis=1)
# clf_gini.fit(X_weighted_corr_train, y_train)

# #Adding more weightage to a specific column
# X_test_scaled = scaler.transform(X_test)
# X_corr_test = (X_test[:, featurelen-5] > 7000).astype(int)[:, np.newaxis]
# X_weighted_test = X_test_scaled * weights
# X_weighted_corr_test = np.concatenate((X_corr_test, X_weighted_test), axis=1)
# result = clf_gini.predict(X_weighted_corr_test)

# # Results after putting weights: 
# # Accuracy: 0.8537559117990295
##############################################################################

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
print ("[Result] Score: " + str(dc.score(X_test,y_test)))
print ("[Result] " + str(ctr) + " of " + str(len(result)) + " (<= 50k: " + str(ctr_true) + ", >50k: " + str(ctr_false) + ")")

# calculate precision and recall scores
accuracy = accuracy_score(y_test, result)
precision = precision_score(y_test, result, average='weighted')
recall = recall_score(y_test, result, average='weighted')

print("[Result] Accuracy: " + str(accuracy))
print("[Result] Precision: " + str(precision))
print("[Result] Recall: " + str(recall))
