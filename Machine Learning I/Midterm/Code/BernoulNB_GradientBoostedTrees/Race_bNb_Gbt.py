#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 10:22:25 2017

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


#Read in dataframe, randomly sampling 100,000 values and dropping 
#HOSPID, KEY, and NISSTRATUM which are noisey variables
df = pd.read_csv("data_train.csv").dropna(1, thresh= 445000).dropna(0, how="any").sample(n=100000, random_state = 12).drop(["HOSPID", "KEY", "NIS_STRATUM"], axis=1)

#==============================================================================
# Bernouli Naive Bayes Classifier as Baseline with Binary Features
#==============================================================================

#Select binary features from df
dfBernouli = df[["RACE", "AWEEKEND", "DIED","FEMALE", "ORPROC", 'CM_AIDS', 'CM_ALCOHOL', 'CM_ANEMDEF',
       'CM_ARTH', 'CM_BLDLOSS', 'CM_CHF', 'CM_CHRNLUNG', 'CM_COAG',
       'CM_DEPRESS', 'CM_DM', 'CM_DMCX', 'CM_DRUG', 'CM_HTN_C', 'CM_HYPOTHY',
       'CM_LIVER', 'CM_LYMPH', 'CM_LYTES', 'CM_METS', 'CM_NEURO', 'CM_OBESE',
       'CM_PARA', 'CM_PERIVASC', 'CM_PSYCH', 'CM_PULMCIRC', 'CM_RENLFAIL',
       'CM_TUMOR', 'CM_ULCER', 'CM_VALVE', 'CM_WGHTLOSS']]

#Convert non-numeric columns to type int
toInt = ["AWEEKEND", "DIED","FEMALE"]
for col in toInt:
    dfBernouli.loc[:, col] = dfBernouli.loc[:, col].astype(int)


XraceBnb = np.array(dfBernouli.loc[:, dfBernouli.columns != 'RACE'])
yRaceBnb = np.array(dfBernouli.loc[:,"RACE"])

#Train test split
X_trainBnb, X_testBnb, y_trainBnb, y_testBnb = train_test_split(XraceBnb, yRaceBnb, test_size=0.33, random_state=42)

#Look at distribution to see class imbalance and print out class
#imbalance with Counter
c = Counter(y_trainBnb)
print('Original dataset shape {}'.format(Counter(y_trainBnb)))
print([(i, c[i] / len(y_trainBnb) * 100.0) for i, count in c.most_common()])
plt.hist(y_trainBnb)
plt.title("Multiclass Distribution of Race")

#Begin Pipeline Setup with a step for univariate feature selection
select = SelectKBest()
bNb = BernoulliNB()
steps = [("feature_selection", select), ("bernouli_nb", bNb)]
pipeNb = Pipeline(steps)

paraGridBnb = dict(feature_selection__k=[30,31,32,33])

#Run 3-fold Cross Validated GridSearch
gsBnb = GridSearchCV(pipeNb, param_grid=paraGridBnb, scoring="f1_macro", n_jobs=-1)
gsBnb.fit(X_trainBnb, y_trainBnb)

#Predict using fitted bNb model and print classification report and accuracy score
BnbPreds = gsBnb.predict(X_testBnb)
BnbReport  = classification_report(BnbPreds, y_testBnb)
BnbScore = accuracy_score(BnbPreds, y_testBnb)
print(BnbReport)
print(BnbScore)

#Save Model
bestModelBnb = gsBnb.best_estimator_
joblib.dump(bestModelBnb, 'Race_BnbBestModel.pkl', compress = 9)

#==============================================================================
#Best Parameters {'feature_selection__k': 32}

#             precision    recall  f1-score   support
#
#          1       1.00      0.69      0.82     32823
#          2       0.02      0.55      0.03       177
#          3       0.00      0.00      0.00         0
#          4       0.00      0.00      0.00         0
#          5       0.00      0.00      0.00         0
#          6       0.00      0.00      0.00         0
#
#    avg / total       0.99      0.69      0.81     33000

# Accuracy Score 0.690545454545
#==============================================================================


#==============================================================================
# Gradient Boosted Trees Classifier without Re-sampling
#==============================================================================

#Create new X and y arrays for train test split
XraceGb = np.array(df.loc[:, df.columns != 'RACE'])
yRaceGb = np.array(df.loc[:,"RACE"])

#Initiate pipeline and paramGrid to pass to Gridserach
pipeGbc = Pipeline([
        ("clf", GradientBoostingClassifier())])
    
learning = [0.2]
samplesSplit = [7,20]
paramGridGbc = [{
        "clf__min_samples_split":samplesSplit,
                 "clf__learning_rate":learning}]
    
gsGbc = GridSearchCV(estimator=pipeGbc, param_grid = paramGridGbc,
                 scoring = "f1_micro", cv = 3, n_jobs = -1)

#Train test split
X_trainGb, X_testGb, y_trainGb, y_testGb = train_test_split(XraceGb, yRaceGb, test_size=0.33, random_state=43)

#Fit grid search
gsGbc.fit(X_trainGb, y_trainGb)

gbcPreds = gsGbc.predict(X_testGb)
print(classification_report(y_testGb, gbcPreds))
print(accuracy_score(y_testGb, gbcPreds))

bestModelGb = gsGbc.best_estimator_

#Save best model and review precision,recall, and f1-score
from sklearn.externals import joblib
joblib.dump(bestModelGb, 'Race_GBModel.pkl', compress = 9)
#==============================================================================
# Best Parameters:{'clf__learning_rate': 0.2, 'clf__min_samples_split': 7}
#              precision    recall  f1-score   support
# 
#           1       0.82      0.95      0.88     22746
#           2       0.57      0.46      0.51      5802
#           3       0.54      0.33      0.41      2688
#           4       0.00      0.00      0.00       430
#           5       0.03      0.02      0.03        42
#           6       0.82      0.20      0.33      1292
# 
# avg / total       0.74      0.77      0.74     33000
# 
# 0.768363636364
#==============================================================================

#BELOW ARE STEPS THAT WERE TAKEN TO VALIDATE THE BEST PARAMETERS FOR A 
#BEST ESTIMATOR. VALIDATION CURVES WERE PRODUCED AND ARE AVAILABLE IN SUPPLEMENTARY
#FILES FOLDER

#==============================================================================
# Gradient Boosted Validation Curve min_samples_split
#==============================================================================

from sklearn.model_selection import validation_curve

param_range = [3,10, 20,40]
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
fig1 = plt.figure()
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
plt.legend(loc='center right')
plt.xlabel('Min Samples Split')
plt.ylabel('Accuracy')
plt.show()
fig1.savefig("Race_min_samples", dpi = 300)
# test_mean: array([ 3: 0.76629874,  10: 0.76592558,  20: 0.7669256 ,  40: 0.76664203])

#==============================================================================
# Gradient Boosted Validation Curve max_depth
#==============================================================================

param_range = [3, 7]
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

fig2 = plt.figure()
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
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.show()
fig2.savefig("Race_max_depth", dpi = 300)

#Test mean: array([ 0.76343317,  0.75716423])

#==============================================================================
# Gradient Boosted Validation Curve learning rate
#==============================================================================

param_range = [0.2, 0.5, 1.0]
train_scores, test_scores = validation_curve(estimator= pipelr,
                                             X= X_train,
                                             y = y_train,
                                             param_name= "clf__learning_rate",
                                             param_range = param_range,
                                             cv = 3)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

import matplotlib.pyplot as plt
fig3 = plt.figure()
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
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.show()
fig3.savefig("Race_learningRate", dpi = 300)
#test mean [ 0.76304508,  0.75623917,  0.72835851]

#==============================================================================
# Cross Validation Curve with Cross Val Score
#==============================================================================

from sklearn.model_selection import cross_val_score

pipeGbcVal = Pipeline([
        ("clf", GradientBoostingClassifier(n_estimators = 300, min_samples_split=20, learning_rate= 0.2))])
scores = cross_val_score(estimator=pipeGbcVal, X=X_trainGb, y=y_trainGb, cv=5, n_jobs= -1)    
# scores:  array([ 0.77270353,  0.7319603 ,  0.77404671,  0.77341593,  0.76518883])

pipeGbcVal500 = Pipeline([
        ("clf", GradientBoostingClassifier(n_estimators = 500, min_samples_split=20, learning_rate= 0.2))])
scores500 = cross_val_score(estimator=pipeGbcVal, X=X_trainGb, y=y_trainGb, cv=5, n_jobs= -1)    
# scores500: array([ 0.77106186,  0.72733378,  0.77203194,  0.77341593,  0.77078668])

pipeGbcVal1000 = Pipeline([
        ("clf", GradientBoostingClassifier(n_estimators = 1000, min_samples_split=20, learning_rate= 0.2))])
scores1000 = cross_val_score(estimator=pipeGbcVal, X=X_trainGb, y=y_trainGb, cv=5, n_jobs= -1)    
#scores1000: array([ 0.76636072,  0.72830386,  0.76755466,  0.7731174 ,  0.77011494])
    

import matplotlib.pyplot as plt
figGbCv = plt.figure()
plt.plot(np.arange(1,6), scores, marker='o', markersize=5,
         label='GBT (n_estimators = 300)')


plt.plot(np.arange(1,6), scores500, marker='o', markersize=5,
         label='GBT (n_estimators = 500)')


plt.plot(np.arange(1,6), scores100, marker='o', markersize=5,
         label='GBT (n_estimators = 1000)')
plt.ylim(0.720, 0.780)
plt.grid()
plt.legend(loc='center right')
plt.xlabel('Cross Validation Iteration')
plt.ylabel('Cross Validated Accuracy: Race')
plt.title("Gradient Boosted Trees: 5-fold Cross Validation")
plt.show()
figGbCv.savefig("Race_GbtCv", dpi = 300)

