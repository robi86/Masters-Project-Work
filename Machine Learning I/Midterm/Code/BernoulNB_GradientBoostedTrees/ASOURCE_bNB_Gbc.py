#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:23:31 2017

@author: davidrobison
"""

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

df = pd.read_csv("data_train.csv").dropna(1, thresh= 445000).dropna(0, how="any").sample(n=100000, random_state = 14).drop(["HOSPID", "KEY", "NIS_STRATUM"], axis=1)
#"ASOURCE", "ATYPE", "TOTCHG", "ZIPINC_QRTL"


#==============================================================================
# Bernouli Naive Bayes Classifier as Baseline with Binary Features
#==============================================================================

dfBernouli = df[["ASOURCE", "AWEEKEND", "DIED","FEMALE", "ORPROC", 'CM_AIDS', 'CM_ALCOHOL', 'CM_ANEMDEF',
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
XsourceBnb = np.array(dfBernouli.loc[:, dfBernouli.columns != 'ASOURCE'])
ySourceBnb = np.array(dfBernouli.loc[:,"ASOURCE"])

X_trainBnb, X_testBnb, y_trainBnb, y_testBnb = train_test_split(XsourceBnb, ySourceBnb, test_size=0.33, random_state=42)

#Look at distribution to see class imbalance and print out class
#imbalance with Counter
c = Counter(y_trainBnb)
print('Original dataset shape {}'.format(Counter(y_trainBnb)))
print([(i, c[i] / len(y_trainBnb) * 100.0) for i, count in c.most_common()])
plt.hist(y_trainBnb)
plt.title("Multiclass Distribution of Race")

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

# Counter({5: 91049, 1: 34008, 2: 4810, 3: 4109, 4: 24})
# Percentages: (5, 67.94701492537314), (1, 25.379104477611943), (2, 3.58955223880597), (3, 3.066417910447761), (4, 0.01791044776119403)]

#               precision    recall  f1-score   support
# 
#           1       0.11      0.35      0.17      5207
#           2       0.01      0.26      0.03       137
#           3       0.05      0.14      0.07       700
#           4       0.00      0.00      0.00         0
#           5       0.93      0.69      0.79     59956
# 
# avg / total       0.85      0.66      0.73     66000
# 
# 
# Accuracy Score: 0.65746969697
#==============================================================================
bestModelBnb = gsBnb.best_estimator_
joblib.dump(bestModelBnb, 'SOURCE_BnbBestModel.pkl', compress = 9)

#==============================================================================
# Gradient Boosting Classifier without Re-sampling
#==============================================================================

XsourceGb = np.array(df.loc[:, df.columns != 'ASOURCE'])
ysourceGb = np.array(df.loc[:,"ASOURCE"])

pipeGbc = Pipeline([
        ("clf", GradientBoostingClassifier())])

estimators = [50, 100]
learningRate = [0.1, 0.2, 0.3, 0.4]

paramGridGbc = [{
        "clf__n_estimators":estimators,
                 "clf__learning_rate":learningRate}]
    
gsGbc = GridSearchCV(estimator=pipeGbc, param_grid = paramGridGbc,
                 scoring = "f1_micro", cv = 3, n_jobs = -1)

X_trainGb, X_testGb, y_trainGb, y_testGb = train_test_split(XsourceGb, ysourceGb, test_size=0.33, random_state=43)

gsGbc.fit(X_trainGb, y_trainGb)

gbcPreds = gsGbc.predict(X_testGb)
print(classification_report(y_testGb, gbcPreds))
print(accuracy_score(y_testGb, gbcPreds))

bestModelGb = gsGbc.best_estimator_
joblib.dump(bestModelGb, 'SOURCE_GbBestModel.pkl', compress = 9)


#   Best Parameters: {'clf__learning_rate': 0.4, 'clf__n_estimators': 100}
#                 precision    recall  f1-score   support
# 
#           1       0.91      0.96      0.93      8402
#           2       0.75      0.56      0.64      1185
#           3       0.61      0.29      0.39      1101
#           4       0.25      0.11      0.15         9
#           5       0.95      0.97      0.96     22303
# 
# avg / total       0.92      0.93      0.92     33000
# 
# Accuracy Score: 0.926909090909
#==============================================================================


#Begin Validation Curve for Parameter Tuning 

df = pd.read_csv("data_train.csv").dropna(1, thresh= 445000).dropna(0, how="any").sample(n=100000, random_state = 11).drop(["HOSPID", "KEY", "NIS_STRATUM"], axis=1)
#"ASOURCE", "ATYPE", "TOTCHG", "ZIPINC_QRTL"

'''Create predictor and response numpy arrays using Pandas indexing for optimal 
performance'''
Xsource = np.array(df.loc[:, df.columns != 'ASOURCE'])
ySource = np.array(df.loc[:,"ASOURCE"])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xsource, ySource, test_size=0.33, random_state=42)

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
fig1source = plt.figure()
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
plt.title("ASOURCE Validation Curve: min_samples_split")
plt.show()
fig1source.savefig("source_min_samples", dpi = 300)
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
plt.title("ASOURCE Validation Curve: max_depth")
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.show()
fig2.savefig("Source_max_depth", dpi = 300)

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
   


