#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 12:20:54 2017

@author: davidrobison
"""


import pandas as pd
import numpy as np
import random 
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

df = pd.read_csv("data_train.csv").dropna(1, thresh= 445000).dropna(0, how="any").sample(n=100000, random_state = 16).drop(["HOSPID", "KEY", "NIS_STRATUM"], axis=1)
#"ASOURCE", "ATYPE", "TOTCHG", "ZIPINC_QRTL"


#==============================================================================
# Bernouli Naive Bayes Classifier as Baseline with Binary Features
#==============================================================================

dfBernouli = df[["ZIPINC_QRTL", "AWEEKEND", "DIED","FEMALE", "ORPROC", 'CM_AIDS', 'CM_ALCOHOL', 'CM_ANEMDEF',
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
XzipBnb = np.array(dfBernouli.loc[:, dfBernouli.columns != 'ZIPINC_QRTL'])
yZipBnb = np.array(dfBernouli.loc[:,"ZIPINC_QRTL"])

X_trainBnb, X_testBnb, y_trainBnb, y_testBnb = train_test_split(XzipBnb, yZipBnb, test_size=0.33, random_state=42)

#Look at distribution to see class imbalance and print out class
#imbalance with Counter
c = Counter(y_trainBnb)
print('Original dataset shape {}'.format(Counter(y_trainBnb)))
print([(i, c[i] / len(y_trainBnb) * 100.0) for i, count in c.most_common()])
plt.hist(y_trainBnb)
plt.title("Multiclass Distribution of ZIPINC_QRTL")

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

bestModelBnb = gsBnb.best_estimator_
from sklearn.externals import joblib
joblib.dump(bestModelGb, 'ZIP_BnbBestModel.pkl', compress = 9)

#==============================================================================
#If I did a feature selection, I believe that the feature removal due to Bnb would have
# higher feature importances. Will need to return. 
#              precision    recall  f1-score   support
# 
#           1       0.35      0.33      0.34      9042
#           2       0.00      0.11      0.00        18
#           3       0.01      0.31      0.02       282
#           4       0.78      0.36      0.49     23658
# 
# avg / total       0.65      0.35      0.45     33000
# 
# 
# Accuracy Score: 0.352212121212
#==============================================================================


#==============================================================================
# Gradient Boosting Classifier without Re-sampling
#==============================================================================

XzipGb = np.array(df.loc[:, df.columns != 'ZIPINC_QRTL'])
yZipGb = np.array(df.loc[:,"ZIPINC_QRTL"])

pipeGbc = Pipeline([
        ("clf", GradientBoostingClassifier())])

learningRate = [0.2]
minSamples = [5,10]

paramGridGbc = [{
        "clf__min_samples_split":minSamples,
                 "clf__learning_rate":learningRate}]
    
gsGbc = GridSearchCV(estimator=pipeGbc, param_grid = paramGridGbc,
                 scoring = "f1_micro", cv = 3, n_jobs = -1)

X_trainGb, X_testGb, y_trainGb, y_testGb = train_test_split(XzipGb, yZipGb, test_size=0.33, random_state=43)

gsGbc.fit(X_trainGb, y_trainGb)

gbcPreds = gsGbc.predict(X_testGb)
print(classification_report(y_testGb, gbcPreds))
print(accuracy_score(y_testGb, gbcPreds))

bestModelGb = gsGbc.best_estimator_

from sklearn.externals import joblib
joblib.dump(bestModelGb, 'ZIP_GbBestModel.pkl', compress = 9)

#==============================================================================
# best_params: {'clf__learning_rate': 0.2, 'clf__min_samples_split': 10}
#           precision    recall  f1-score   support
# 
#           1       0.64      0.70      0.67      8394
#           2       0.49      0.40      0.44      5833
#           3       0.48      0.39      0.43      7627
#           4       0.69      0.79      0.74     11146
# 
# avg / total       0.59      0.61      0.60     33000
# 
# Accuracy Score: 0.606393939394
#==============================================================================

#==============================================================================
# Gradient Boosting Classifier with Over and Under Resampling
#==============================================================================



pipelr.fit(X_train, y_train)

print("test accuracy: %.3f" %pipelr.score(X_test, y_test))



n