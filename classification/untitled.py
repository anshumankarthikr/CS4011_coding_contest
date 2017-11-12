from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from mlxtend.feature_extraction import PrincipalComponentAnalysis as PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

X = np.genfromtxt('../../contest_data/xtrain_linear_imputed.csv', delimiter=',')
print 'loaded X'
y = np.genfromtxt('../../contest_data/train.csv', delimiter=',')[1:,-1]
print 'loaded y'


gnb = GaussianNB()
bag=BaggingClassifier(base_estimator=gnb, n_estimators=1000, max_samples=1.0, 
	max_features=0.2, bootstrap=True, bootstrap_features=False, oob_score=False, 
	warm_start=False, n_jobs=1, random_state=0, verbose=1)
scores = cross_val_score(bag, X, y,scoring='f1_micro',cv=5,verbose=5)
print scores.mean()