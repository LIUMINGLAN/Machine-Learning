# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import StratifiedKFold,cross_validate,cross_val_score

Xtr=np.loadtxt('./Xtrain.csv')
ytr=np.loadtxt('./Ytrain.csv')
Xtt=np.loadtxt('./Xtest.csv')

#k_range = range(1, 10)
#k_scores = pd.DataFrame(index =['test_accuracy','test_roc_auc','test_average_precision'] )
#
#for k in k_range:
#    knn = KNN(n_neighbors = k)
#    scores = cross_validate(knn, Xtr, ytr, cv=5, scoring=['accuracy','roc_auc','average_precision'])
#    k_score = pd.DataFrame(scores).mean()
#    k_scores = pd.merge(k_scores, pd.DataFrame(k_score), left_index=True, right_index=True)
#
#print(k_scores)
#k_scores=k_scores.T

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
rf_params = {
    'C': [1,10],
    "kernel":['linear','rbf']
}
clf = SVC(gamma='scale')
grid = GridSearchCV(clf, rf_params, cv=5, scoring='accuracy')
grid.fit(Xtr, ytr)
print(grid.best_params_)
print("Accuracy:"+ str(grid.best_score_))