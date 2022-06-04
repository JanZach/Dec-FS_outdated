# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 13:13:20 2021

@author: janmo
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,precision_score, recall_score, roc_auc_score, confusion_matrix, average_precision_score
from xgboost import XGBClassifier
from xgboost import plot_importance

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import shap
from imblearn.over_sampling import RandomOverSampler,SMOTE
from sklearn.ensemble import RandomForestClassifier
import random
import statistics
import pickle

Data = pd.read_csv('C:/Users/janmo/OneDrive/Dokumente/Goethe Uni/Module/Master Thesis/XAI/german_credit.csv')

X = Data.iloc[:,0:-1]                   # Initialize X variables
X = X.iloc[:,0:6]                       # reduce X to 5

X_num_vars = ["duration", "amount"]
X_nom_vars = ["purpose"]
X_ord_vars = ["status", "savings"]

X_num = X.loc[:,X_num_vars]         # Assign numeric vars
X_nom = X.loc[:,X_nom_vars]         # Assign nominal vars
X_ord = X.loc[:,X_ord_vars]         # Assign ordinal vars

# Ordinal- and nominal encoder
# -- ordinal -- #
ordinal_encoder = OrdinalEncoder(categories=[['... < 0 DM', 'no checking account', '0<= ... < 200 DM', '... >= 200 DM / salary for at least 1 year'],
                                             ['unknown/no savings account', '... <  100 DM', '100 <= ... <  500 DM', '500 <= ... < 1000 DM', '... >= 1000 DM']])
X_ord_encoded   = ordinal_encoder.fit_transform(X_ord)
X_ord_encoded   = pd.DataFrame(X_ord_encoded, columns = X_ord_vars)

# -- nominal -- #
nom_encoder     = OneHotEncoder()
X_nom_encoded   = nom_encoder.fit_transform(X_nom)
nom_encoder.categories_ # what are the transformed dummy variables?
X_nom_encoded_vars = ['business', 'car (new)', 'car (used)', 'domestic appliances',
       'furniture/equipment', 'others', 'radio/television', 'repairs',
       'retraining', 'vacation']
X_nom_encoded   = X_nom_encoded.toarray()   # convert onehot sparse matrix to dense matrix
X_nom_encoded   = pd.DataFrame(X_nom_encoded, columns = X_nom_encoded_vars)

# concatenate numerical, ordinal, and nominal attributes
X_encoded = X_num.join(X_ord_encoded).join(X_nom_encoded)


##### For oTree __init__ file: get list of all medians
list_medians = []
for col in X_encoded.columns:
    median = X_encoded.loc[:,col].median()
    list_medians.append(median)
list_medians

# Create the dummies via for-loop
X_encoded_columns = X_encoded.columns
for column in X_encoded_columns:
    X_encoded[column+"_dummy"] = 0

#randomly insert nan
for col in X_encoded.columns[0:14]:
    X_encoded.loc[X_encoded.sample(frac=0.1).index, col] = np.nan

#Process the nan's
for col in range(0,14):
    for i in range(0,len(X_encoded)):
        if np.isnan(X_encoded.iloc[i,col]):
            X_encoded.iloc[i, col+14] = 1

# na imputer
imputer = SimpleImputer(strategy="median")
imputer.fit(X_encoded)
X_encoded_imputed = imputer.transform(X_encoded)
X_encoded_imputed = pd.DataFrame(X_encoded_imputed, columns=X_encoded.columns)


# prepare the target variable
y = Data.iloc[:,-1]
y.replace({'bad':0, 'good':1}, inplace=True)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded_imputed,y,
                                                    test_size=0.3,
                                                    random_state=12345)

# Optionally: perform SMOTE
""""    
smote                         = SMOTE(sampling_strategy='minority')
X_train_smote, y_train_smote  = smote.fit_resample(X_train, y_train)
"""

# Train an XGBoost model
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss').fit(X_train, y_train)


# ------------- Test the functionality --------------- #

# Create an test instance for model prediction
test_obs = X_encoded.iloc[1,0:14]
test_obs = pd.DataFrame(test_obs).T

# insert an nan
test_obs.iloc[0, [0,3,5,8]] = np.nan

# Create the dummies via for-loop
test_obs_columns = test_obs.columns
for column in test_obs_columns:
    test_obs[column+"_dummy"] = 0

#Process the nan's
for col in range(0,14):
    if np.isnan(test_obs.iloc[0,col]):
        test_obs.iloc[0, col+14] = 1

# Impute nas
test_obs_imputed = imputer.transform(test_obs)
test_obs_imputed = pd.DataFrame(test_obs_imputed, columns=test_obs.columns)

# create prediction
pred = xgb_clf.predict(test_obs_imputed)





"""
for i in range(0, 3):
    if np.isnan(input_obs.iloc[0,i]):
        input_obs.iloc[0, i+3]=1
        impute = "feature" +str(i)+"_median"
        input_obs.iloc[0,i] = locals()[impute]

xgb_clf.predict(input_obs)


# save model
#filename = 'C:/Users/janmo/OneDrive/Dokumente/Goethe Uni/Doktor/Projekte/Decentralized Feature Selection 1/xgboost_otree_test_v2_impute.sav'
#pickle.dump(xgb_clf, open(filename, 'wb'))

"""