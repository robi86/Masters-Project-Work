#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 10:48:38 2017

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

df = pd.read_csv("data_train.csv").dropna(1, thresh= 445000).dropna(0, how="any").sample(n=100000, random_state = 16).drop(["HOSPID", "KEY", "NIS_STRATUM"], axis=1)
#"ASOURCE", "ATYPE", "TOTCHG", "ZIPINC_QRTL"

labels = ['1', '2', '3', '4', '5'] #just class labels we'll call 1-5

bins = [0, 1000, 5000, 10000, 20000, 1500000]

df = df.assign(TOTCHG = pd.cut(df.TOTCHG, bins, labels=labels)) #add a new column with our cuts

#==============================================================================
# Bernouli Naive Bayes Classifier as Baseline with Binary Features
#==============================================================================


dfBernouli = df[["TOTCHG", "AWEEKEND", "DIED","FEMALE", "ORPROC", 'CM_AIDS', 'CM_ALCOHOL', 'CM_ANEMDEF',
       'CM_ARTH', 'CM_BLDLOSS', 'CM_CHF', 'CM_CHRNLUNG', 'CM_COAG',
       'CM_DEPRESS', 'CM_DM', 'CM_DMCX', 'CM_DRUG', 'CM_HTN_C', 'CM_HYPOTHY',
       'CM_LIVER', 'CM_LYMPH', 'CM_LYTES', 'CM_METS', 'CM_NEURO', 'CM_OBESE',
       'CM_PARA', 'CM_PERIVASC', 'CM_PSYCH', 'CM_PULMCIRC', 'CM_RENLFAIL',
       'CM_TUMOR', 'CM_ULCER', 'CM_VALVE', 'CM_WGHTLOSS']]

toInt = ["AWEEKEND", "DIED","FEMALE"]

for col in toInt:
    dfBernouli.loc[:, col] = dfBernouli.loc[:, col].astype(int)

from sklearn.preprocessing import LabelEncoder
labelencoder =LabelEncoder()    
dfBernouli.loc[:,"TOTCHG"] = labelencoder.fit_transform(dfBernouli.loc[:,"TOTCHG"].cat.codes)

'''Create predictor and response numpy arrays using Pandas indexing for optimal 
performance'''
XtotchgBnb = np.array(dfBernouli.loc[:, dfBernouli.columns != 'TOTCHG'])
yTotchgBnb = np.array(dfBernouli.loc[:,"TOTCHG"])

X_trainBnb, X_testBnb, y_trainBnb, y_testBnb = train_test_split(XtotchgBnb, yTotchgBnb, test_size=0.33, random_state=42)

#Look at distribution to see class imbalance and print out class
#imbalance with Counter
c = Counter(y_trainBnb)
print('Original dataset shape {}'.format(Counter(y_trainBnb)))
print([(i, c[i] / len(y_trainBnb) * 100.0) for i, count in c.most_common()])
plt.hist(y_trainBnb)
plt.title("Multiclass Distribution of TOTCHG")

#Begin Pipeline Setup
select = SelectKBest()
bNb = BernoulliNB()
steps = [("feature_selection", select), ("bernouli_nb", bNb)]
pipeNb = Pipeline(steps)

paraGridBnb = dict(feature_selection__k=[15, 20, 25])

gsBnb = GridSearchCV(pipeNb, param_grid=paraGridBnb, scoring="f1_macro", n_jobs=-1)

gsBnb.fit(X_trainBnb, y_trainBnb)

BnbPreds = gsBnb.predict(X_testBnb)

BnbReport  = classification_report(BnbPreds, y_testBnb)
BnbScore = accuracy_score(BnbPreds, y_testBnb)
print(BnbReport)
print(BnbScore)

bestModelBnb = gsBnb.best_estimator_
joblib.dump(bestModelBnb, 'TOTCHG_BnbBestModel.pkl', compress = 9)

#==============================================================================
# Best Params: {'feature_selection__k': 25}
# Distribution:  [(4, 32.88208955223881), (3, 24.6955223880597), (2, 24.459701492537313), 
#  (1, 17.313432835820898), (0, 0.6492537313432836)]
# 
#              precision    recall  f1-score   support
# 
#           0       0.00      0.00      0.00         0
#           1       0.45      0.36      0.40      7065
#           2       0.30      0.33      0.32      7313
#           3       0.16      0.30      0.21      4328
#           4       0.68      0.52      0.59     14294
# 
# avg / total       0.48      0.42      0.44     33000
# 
# Accuracy Score: 0.415848484848
#==============================================================================


#==============================================================================
# Gradient Boosting Classifier without Re-sampling
#==============================================================================

XtotchgGb = np.array(df.loc[:, df.columns != 'TOTCHG'])
yTotchgGb = np.array(df.loc[:,"TOTCHG"])

pipeGbc = Pipeline([
        ("clf", GradientBoostingClassifier())])

learning = [0.1]
samplesSplit = [30,50]

paramGridGbc = [{
        "clf__min_samples_split":samplesSplit,
                 "clf__learning_rate":learning}]
    
gsGbc = GridSearchCV(estimator=pipeGbc, param_grid = paramGridGbc,
                 scoring = "f1_micro", cv = 3, n_jobs = -1)

X_trainGb, X_testGb, y_trainGb, y_testGb = train_test_split(XtotchgGb, yTotchgGb, test_size=0.33, random_state=43)

gsGbc.fit(X_trainGb, y_trainGb)

gbcPreds = gsGbc.predict(X_testGb)
print(classification_report(y_testGb, gbcPreds))
print(accuracy_score(y_testGb, gbcPreds))

bestModelGb = gsGbc.best_estimator_

from sklearn.externals import joblib
joblib.dump(bestModelGb, 'TOTCHG_GBModel.pkl', compress = 9)

#==============================================================================
# Best Params: {'clf__learning_rate': 0.2, 'clf__min_samples_split': 30}
#              precision    recall  f1-score   support
# 
#           1       0.81      0.52      0.64       232
#           2       0.75      0.73      0.74      5627
#           3       0.63      0.70      0.66      8025
#           4       0.63      0.57      0.60      8162
#           5       0.85      0.86      0.86     10954
# 
# avg / total       0.73      0.73      0.72     33000
# 
# Accuracy Score: 0.725757575758
#==============================================================================
