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
import  pickle

X = np.genfromtxt('../../contest_data/xtrain_linear_imputed.csv', delimiter=',')
print 'loaded X'
y = np.genfromtxt('../../contest_data/train.csv', delimiter=',')[1:,-1]
print 'loaded y'



et = ExtraTreesClassifier(n_estimators=300, max_depth=None, random_state=0,verbose=1)
bag=BaggingClassifier(base_estimator=et, n_estimators=20, max_samples=1.0, 
	max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, 
	warm_start=False, n_jobs=1, random_state=0, verbose=1)
scores = cross_val_score(bag, X, y,scoring='f1_micro',cv=5,verbose=5)
print scores.mean()
'''
trees 	estims 	f1
300		10		33.4
300		20		
150		50		
'''



pca = PCA(n_components=1000)
X_pca = pca.fit(X).transform(X)

et = ExtraTreesClassifier(n_estimators=300, max_depth=None, random_state=0,verbose=1)
bag=BaggingClassifier(base_estimator=et, n_estimators=20, max_samples=1.0, 
	max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, 
	warm_start=False, n_jobs=1, random_state=0, verbose=1)
scores = cross_val_score(bag, X_pca, y,scoring='f1_micro',cv=5,verbose=5)
print scores.mean()
'''
trees 	estims 	f1
300		20		31
300		10		30	

'''

for tree in [150,300,500,1000]:
	for est in [10,20,50,100,200,1000]:

		et = ExtraTreesClassifier(n_estimators=tree, max_depth=None, random_state=0,verbose=1)
		bag=BaggingClassifier(base_estimator=et, n_estimators=est, max_samples=1.0, 
			max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, 
			warm_start=False, n_jobs=1, random_state=0, verbose=1)
		scores = cross_val_score(bag, X, y,scoring='f1_micro',cv=5,verbose=5)
		print 'number of trees',tree
		print 'number of bags=',est
		print 'f1 scores=',scores
		print 'mean f1=',scores.mean()
		print 'std dev f1= ',scores.std()
		data=np.genfromtxt('bagging_et/metrics.csv',delimiter=',')
		newrow=np.hstack(([tree,est],scores,scores.mean(),scores.std()))
		data=np.vstack((data,newrow))
		np.savetxt('bagging_et/metrics.csv',data,delimiter=',')
		pickle.dump(bag,open('bagging_et/tr'+str(tree)+'bg'+str(est)+'.pkl','w'))





svc = SVC(C=100)
bag=BaggingClassifier(base_estimator=svc, n_estimators=50, max_samples=1.0, 
	max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, 
	warm_start=True, n_jobs=-1, random_state=0, verbose=1)
scores = cross_val_score(bag, X, y,scoring='f1_micro',cv=5,verbose=5)
print scores.mean()

'''
C 		estims 		f1
1000	50			35.7
100		100			36
100		50			35.3
'''


pca = PCA(n_components=1400)
X_pca = pca.fit(X).transform(X)
svc = SVC(C=1000)
bag=BaggingClassifier(base_estimator=svc, n_estimators=50, max_samples=1.0, 
	max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, 
	warm_start=False, n_jobs=-1, random_state=0, verbose=1)
scores = cross_val_score(bag, X_norm, y,scoring='f1_micro',cv=4,verbose=5,n_jobs=-1)
print scores.mean()

a=[]
for i in [10,20,50,100,500]:
	gnb = GaussianNB()
	bag=BaggingClassifier(base_estimator=gnb, n_estimators=20, max_samples=1.0, 
		max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, 
		warm_start=False, n_jobs=1, random_state=0, verbose=1)
	scores = cross_val_score(bag, X, y,scoring='f1_micro',cv=4,verbose=5,n_jobs=-1)
	print scores.mean()
	a.append(scores.mean())
plt.plot([10,20,50,100,500],a)
plt.xlabel('number of estimators')
plt.ylabel('f1')
plt.title('number of estimators in bagging vs f1 score')
plt.savefig('bayes_bags.png')



