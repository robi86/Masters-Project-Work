#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:58:16 2017

@author: davidrobison
"""


import pandas as pd
import numpy as np
import random 
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

df = pd.read_csv("data_train.csv").dropna(1, thresh= 445000).dropna(0, how="any").sample(n=100000, random_state = 15).drop(["HOSPID", "KEY", "NIS_STRATUM"], axis=1)
#"ASOURCE", "ATYPE", "TOTCHG", "ZIPINC_QRTL"


#==============================================================================
# Bernouli Naive Bayes Classifier as Baseline with Binary Features
#==============================================================================

dfBernouli = df[["ATYPE", "AWEEKEND", "DIED","FEMALE", "ORPROC", 'CM_AIDS', 'CM_ALCOHOL', 'CM_ANEMDEF',
       'CM_ARTH', 'CM_BLDLOSS', 'CM_CHF', 'CM_CHRNLUNG', 'CM_COAG',
       'CM_DEPRESS', 'CM_DM', 'CM_DMCX', 'CM_DRUG', 'CM_HTN_C', 'CM_HYPOTHY',
       'CM_LIVER', 'CM_LYMPH', 'CM_LYTES', 'CM_METS', 'CM_NEURO', 'CM_OBESE',
       'CM_PARA', 'CM_PERIVASC', 'CM_PSYCH', 'CM_PULMCIRC', 'CM_RENLFAIL',
       'CM_TUMOR', 'CM_ULCER', 'CM_VALVE', 'CM_WGHTLOSS']]

toInt = ["AWEEKEND", "DIED","FEMALE"]

for col in toInt:
    dfBernouli.loc[:, col] = dfBernouli.loc[:, col].astype(int)

'''Create predictor and response numpy arrays using Pandas indexing for optimal 
performance'''
XtypeBnb = np.array(dfBernouli.loc[:, dfBernouli.columns != 'ATYPE'])
yTypeBnb = np.array(dfBernouli.loc[:,"ATYPE"])

X_trainBnb, X_testBnb, y_trainBnb, y_testBnb = train_test_split(XtypeBnb, yTypeBnb, test_size=0.33, random_state=42)

#Look at distribution to see class imbalance and print out class
#imbalance with Counter
c = Counter(y_trainBnb)
print('Original dataset shape {}'.format(Counter(y_trainBnb)))
print([(i, c[i] / len(y_trainBnb) * 100.0) for i, count in c.most_common()])
plt.hist(y_trainBnb)
plt.title("Multiclass Distribution of Type")

#Begin Pipeline Setup
select = SelectKBest()
bNb = BernoulliNB()
steps = [("feature_selection", select), ("bernouli_nb", bNb)]
pipeNb = Pipeline(steps)

paraGridBnb = dict(feature_selection__k=[20,25,30])

gsBnb = GridSearchCV(pipeNb, param_grid=paraGridBnb, scoring="f1_micro", n_jobs=-1)

gsBnb.fit(X_trainBnb, y_trainBnb)

BnbPreds = gsBnb.predict(X_testBnb)

BnbReport  = classification_report(BnbPreds, y_testBnb)
BnbScore = accuracy_score(BnbPreds, y_testBnb)
BnbMatrix = confusion_matrix(BnbPreds, y_testBnb)

# Counter({1: 35955, 3: 15181, 2: 12264, 4: 3397, 6: 105, 5: 98})
# Percentages: [(1, 53.66417910447762), (3, 22.65820895522388), (2, 18.3044776119403), (4, 5.070149253731343), (6, 0.15671641791044777), (5, 0.1462686567164179)]
#              precision    recall  f1-score   support
#
#          1       0.82      0.70      0.75     20770
#          2       0.17      0.34      0.23      3020
#          3       0.56      0.60      0.58      6976
#          4       0.25      0.19      0.21      2233
#          5       0.00      0.00      0.00         0
#          6       0.00      0.00      0.00         1
#
#avg / total       0.67      0.61      0.63     33000

# Accuracy Score: 0.608363636364
#==============================================================================

bestModelBnb = gsBnb.best_estimator_
joblib.dump(bestModelBnb, 'Type_BnbBestModel.pkl', compress = 9)

#==============================================================================
# Gradient Boosting Classifier without Re-sampling
#==============================================================================

XtypeGb = np.array(df.loc[:, df.columns != 'ATYPE'])
yTypeGb = np.array(df.loc[:,"ATYPE"])

pipeGbc = Pipeline([
        ("clf", GradientBoostingClassifier())])

estimators = [50, 100]
learningRate = [0.1, 0.2, 0.3]

paramGridGbc = [{
        "clf__n_estimators":estimators,
                 "clf__learning_rate":learningRate}]
    
gsGbc = GridSearchCV(estimator=pipeGbc, param_grid = paramGridGbc,
                 scoring = "f1_micro", cv = 3, n_jobs = -1)

X_trainGb, X_testGb, y_trainGb, y_testGb = train_test_split(XtypeGb, yTypeGb, test_size=0.33, random_state=43)

gsGbc.fit(X_trainGb, y_trainGb)

gbcPreds = gsGbc.predict(X_testGb)
print(classification_report(y_testGb, gbcPreds))
print(accuracy_score(y_testGb, gbcPreds))

#              precision    recall  f1-score   support
# 
#           1       0.88      0.94      0.91     17699
#           2       0.81      0.69      0.75      6006
#           3       0.85      0.83      0.84      7527
#           4       1.00      1.00      1.00      1667
#           5       0.72      0.81      0.76        47
#           6       0.86      0.57      0.69        54
# 
# avg / total       0.87      0.87      0.87     33000
# 
# Accuracy Score 0.871242424242
# Best Params: {'clf__learning_rate': 0.3, 'clf__n_estimators': 100}

bestModelGb = gsGbc.best_estimator_
joblib.dump(bestModelGb, 'Type_GbBestModel.pkl', compress = 9)

#==============================================================================
# Cross Validated Gradient Boosting 
#==============================================================================


#Begin Validation Curve for Parameter Tuning 

df = pd.read_csv("data_train.csv").dropna(1, thresh= 445000).dropna(0, how="any").sample(n=100000, random_state = 11).drop(["HOSPID", "KEY", "NIS_STRATUM"], axis=1)
#"ASOURCE", "ATYPE", "TOTCHG", "ZIPINC_QRTL"

'''Create predictor and response numpy arrays using Pandas indexing for optimal 
performance'''
Xtype = np.array(df.loc[:, df.columns != 'ATYPE'])
ytype = np.array(df.loc[:,"ATYPE"])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xtype, ytype, test_size=0.33, random_state=42)

pipelr = Pipeline([ 
                # ("pca", PCA(n_components=5)), #actually performs worse with PCA 
                  ("clf", GradientBoostingClassifier(learning_rate = 0.2))])
    
from sklearn.model_selection import validation_curve

#==============================================================================
# Gradient Boosted Validation Curve min_samples_split
#==============================================================================

param_range = [3,10, 20]
train_scores, test_scores = validation_curve(estimator= pipelr,
                                             X= X_train,
                                             y = y_train,
                                             param_name= "clf__min_samples_split",
                                             param_range = param_range,
                                             cv = 5)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

import matplotlib.pyplot as plt
fig1type = plt.figure()
plt.plot(param_range, train_mean, marker='o', markersize=5,
         label='training accuracy')
plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15)
plt.plot(param_range, test_mean, linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')
plt.fill_between(param_range, test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15)
plt.grid()
plt.legend(loc='center right')
plt.xlabel('Min Samples Split')
plt.ylabel('Accuracy')
plt.title("ATYPE Validation Curve: min_samples_split")
plt.show()
fig1type.savefig("type_min_samples", dpi = 300)
# test_mean: array([ 0.91811976,  0.91788098,  0.91797046])

#==============================================================================
# Gradient Boosted Validation Curve max_depth
#==============================================================================

param_range = [3, 20]
train_scores, test_scores = validation_curve(estimator= pipelr,
                                             X= X_train,
                                             y = y_train,
                                             param_name= "clf__max_depth",
                                             param_range = param_range,
                                             cv = 3)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

fig2type = plt.figure()
import matplotlib.pyplot as plt
plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5,
         label='training accuracy')
plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')
plt.plot(param_range, test_mean, color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')
plt.fill_between(param_range, test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')
plt.grid()
plt.legend(loc='upper right')
plt.title("ASOURCE Validation Curve: max_depth")
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.show()
fig2type.savefig("Source_max_depth", dpi = 300)

#Test mean: array([ 0.76343317,  0.75716423])


#==============================================================================
# Cross Validation Curve with Cross Val Score
#==============================================================================

pipeGbcVal = Pipeline([
        ("clf", GradientBoostingClassifier(n_estimators = 300, min_samples_split=20, learning_rate= 0.4))])
scores = cross_val_score(estimator=pipeGbcVal, X=X_train, y=y_train, cv=5, n_jobs= -1)    
# scores:  array([ 0.77270353,  0.7319603 ,  0.77404671,  0.77341593,  0.76518883])

pipeGbcVal500 = Pipeline([
        ("clf", GradientBoostingClassifier(n_estimators = 500, min_samples_split=20, learning_rate= 0.4))])
scores500 = cross_val_score(estimator=pipeGbcVal, X=X_train, y=y_train, cv=5, n_jobs= -1)    
# scores500: array([ 0.77106186,  0.72733378,  0.77203194,  0.77341593,  0.77078668])

pipeGbcVal1000 = Pipeline([
        ("clf", GradientBoostingClassifier(n_estimators = 1000, min_samples_split=20, learning_rate= 0.2))])
scores1000 = cross_val_score(estimator=pipeGbcVal, X=X_train, y=y_train, cv=5, n_jobs= -1)    
#scores1000: array([ 0.76636072,  0.72830386,  0.76755466,  0.7731174 ,  0.77011494])


import matplotlib.pyplot as plt
figGbSourceCv = plt.figure()
plt.plot(np.arange(1,6), scores, marker='o', markersize=5,
         label='GBT (n_estimators = 300)')


plt.plot(np.arange(1,6), scores500, marker='o', markersize=5,
         label='GBT (n_estimators = 500)')


plt.plot(np.arange(1,6), scores1000, marker='o', markersize=5,
         label='GBT (n_estimators = 1000)')
plt.grid()
plt.legend(loc='center right')
plt.xlabel('Cross Validation Iteration')
plt.ylabel('Cross Validated Accuracy: ASOURCE')
plt.title("Gradient Boosted Trees: 5-fold Cross Validation")
plt.show()
figGbSourceCv.savefig("SOURCE_GbtCv", dpi = 300)