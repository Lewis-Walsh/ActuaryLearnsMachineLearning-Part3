# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 07:56:24 2020

@author: walshl
"""

#%% Do imports

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance

#%% Load test and train data
        
train_data = pd.read_csv("C:\\Work\\Machine Learning experiments\\Kaggle\\Titanic\\train2.csv")
train_data.head()

test_data = pd.read_csv("C:\\Work\\Machine Learning experiments\\Kaggle\\Titanic\\test2.csv")
test_data.head()

#%% Set-up model

Y = train_data["Survived"]
features = ["Pclass","Sex","SibSp","Parch","Embarked","AgeBand"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
columns = X.columns
ColList = columns.tolist()

#%% Run model
RFmodel = RandomForestClassifier(n_estimators=100,random_state=1)
RFmodel.fit(X,Y)
predictions = RFmodel.predict(X_test)


#%% Get Feature importance
Feature_import = RFmodel.feature_importances_
FIdf = pd.DataFrame(Feature_import,ColList)
ax = FIdf.plot.bar()


#%%Set up space of possible hyper-params

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


#%%

RFmodel = RandomForestClassifier(random_state=1)

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
RFmodel_random = RandomizedSearchCV(estimator = RFmodel, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=1, n_jobs = -1)
# Fit the random search model
RFmodel_random.fit(X, Y)
predictions = RFmodel_random.predict(X_test)
RFmodel_random.best_params_

#%% Output

output = pd.DataFrame({'PassengerID': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission100.csv',index=False)

