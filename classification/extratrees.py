from sklearn.ensemble import ExtraTreesClassifier
from mlxtend.feature_extraction import PrincipalComponentAnalysis as PCA
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
X = np.genfromtxt('../../contest_data/xtrain_linear_imputed.csv', delimiter=',')
y = np.genfromtxt('../../contest_data/train.csv', delimiter=',')[1:,-1]




pca = PCA(n_components=1000)
X_pca = pca.fit(X).transform(X)
et = ExtraTreesClassifier(n_estimators=1000, max_depth=None, random_state=0,verbose=0)
scores = cross_val_score(et, X_pca, y,scoring='f1_micro',cv=5,verbose=5)
print scores.mean()


et = ExtraTreesClassifier(n_estimators=2000, max_depth=None, random_state=0,verbose=1)
scores = cross_val_score(et, X, y,scoring='f1_micro',cv=4,verbose=5,n_jobs=-1)
print scores.mean()
'''
components=1000,estimators=1000 gives 32.6% f1
'''

pca = PCA(n_components=1000)
X_pca = pca.fit(xtrain).transform(xtrain)
et = ExtraTreesClassifier(n_estimators=1000, max_depth=None, random_state=0,verbose=0)
scores = cross_val_score(et, X_pca, ytrain,scoring='f1_micro',cv=5,verbose=5)
print scores.mean()

<<<<<<< HEAD

f1scores=[]
for i in range(100,1001,100):
	et = ExtraTreesClassifier(n_estimators=i, max_depth=None, random_state=0,verbose=1)
	scores = cross_val_score(et, X, y,scoring='f1_micro',cv=4,verbose=5,n_jobs=-1)
	f1scores.append(scores.mean())
plt.plot(range(100,1001,100),f1scores)
plt.xlabel('number of trees')
plt.ylabel('5 fold cross validated f1 score')
plt.title('number of trees in Extratrees vs f1')
plt.savefig('trees_vs_f1_et.png')
=======
#generate testing data

X_test = np.genfromtxt('../../contest_data/xtest_linear_imputed.csv', delimiter=',')
et.fit(X,y)
y_predict = et.predict(X_test)




et,clf,bag,xgboost

meta=svc(C=100)

X_meta=np.hstack((X_scale,et.predict(X_scale),bag.predict(X_scale),clf.predict(X_scale)))

scores = cross_val_score(meta, X_meta, ytrain,scoring='f1_micro',cv=4,verbose=5,n_jobs=-1)


#voting
c=np.zeros((y.shape[0],29))
enc = OneHotEncoder()

for mod in [et,clf,bag,xgb]:
	c+=enc.fit_transform(mod.predict(X_scale))
preds=np.argmax(c,axis=1)




>>>>>>> 49831488d6f088a22fb3538a8b8734a57b6ef1dd
