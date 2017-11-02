import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn import linear_model
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
import pickle

X = np.genfromtxt('../../contest_data/train.csv', delimiter=',')[1:,501:-1]
y = np.genfromtxt('../../contest_data/train.csv', delimiter=',')[1:,-1]

X_full=np.genfromtxt('../../contest_data/train.csv', delimiter=',')[1:,1:-1]
#logreg baseline

#define logistic regression model
logreg = linear_model.LogisticRegression(C=1e5,n_jobs=1,verbose=1)

#recursive feature elimination
rfecv = RFECV(estimator=logreg, step=200, cv=StratifiedKFold(3),
              scoring='f1_micro',verbose=10,n_jobs=1)
rfecv.fit(X, y)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation f1 score")
plt.title('f1 vs dropped features (logreg)')
plt.plot([0]+range(100,2200,200), rfecv.grid_scores_)
#plt.show()
plt.savefig('logreg/logreg_dropped_features.png')

#saving stats
pickle.dump(rfecv,open('logreg/model.pkl','w'))
np.savetxt('logreg/gridscores.csv',rfecv.grid_scores_,delimiter=',')
np.savetxt('logreg/ranking.csv',rfecv.ranking_,delimiter=',')
np.savetxt('logreg/mask.csv',rfecv.support_,delimiter=',')

'''
skf = StratifiedKFold(n_splits=3)
for train_index, test_index in skf.split(X, y):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	logreg.fit(X_train, y_train)
	print f1_score(y_test, logreg.predict(X_test), average='micro')
'''


#linear SVM
svc = SVC(kernel="linear")
rfecv = RFECV(estimator=svc, step=200, cv=StratifiedKFold(3),
              scoring='f1_micro',verbose=1,)
rfecv.fit(X, y)

for train_index, test_index in skf.split(X, y):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	logreg.fit(X_train, y_train)
	print f1_score(y_test, logreg.predict(X_test), average='micro')

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation f1 score")
plt.title('accuracy vs dropped features (linear SVM)')
plt.plot([0]+range(100,2200,200), rfecv.grid_scores_)
#plt.show()
plt.savefig('linsvm/linsvm_dropped_features.png')
pickle.dump(rfecv,open('linsvm/model.pkl','w'))
np.savetxt('linsvm/gridscores.csv',rfecv.grid_scores_,delimiter=',')
np.savetxt('linsvm/ranking.csv',rfecv.ranking_,delimiter=',')
np.savetxt('linsvm/mask.csv',rfecv.support_,delimiter=',')

skf = StratifiedKFold(n_splits=3)
svm=pickle.load(open('linsvm/model.pkl', 'rb'))
for train_index, test_index in skf.split(X, y):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	print f1_score(y_test, svm.predict(X_test),average=None)

#decision trees

from sklearn.tree import DecisionTreeClassifier

#without missing rows
# Fit regression model
dt = DecisionTreeClassifier(min_samples_leaf=5)
rfecv = RFECV(estimator=dt, step=200, cv=StratifiedKFold(3),
              scoring='f1_micro',verbose=1,)
rfecv.fit(X, y)
#15%
#with missing rows
dt = DecisionTreeClassifier(min_samples_leaf=5)
rfecv = RFECV(estimator=dt, step=200, cv=StratifiedKFold(5),
              scoring='f1_micro',verbose=1,)
rfecv.fit(X_full, y)


#random forests
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(rf, X_full, y,scoring='f1_micro',cv=5,verbose=5)
print scores.mean()
#30%

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
scores = cross_val_score(lda, X, y,scoring='f1_micro',cv=5,verbose=5)

print scores
#15%
lda.fit(X,y)
X_lda=lda.transform(X)

svc = SVC(kernel="rbf")

skf = StratifiedKFold(n_splits=5)
for train_index, test_index in skf.split(X, y):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	lda.fit(X_train,y_train)
	X_lda_train=lda.transform(X_train)
	X_lda_test=lda.transform(X_test)
	svc.fit(X_lda_train, y_train)
	print f1_score(y_test, svc.predict(X_lda_test), average='micro')
'''

#QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qda = QuadraticDiscriminantAnalysis()
scores = cross_val_score(qda, X, y,scoring='f1_micro',cv=5,verbose=5)

print scores
#15%

'''

#XGBOOST
from xgboost import XGBClassifier
model = XGBClassifier(objective='f1_micro',silent=False,max_depth=3)
#scores = cross_val_score(model, X_full, y,scoring='f1_micro',cv=5,verbose=5)
scores=[]
skf = StratifiedKFold(n_splits=5)
for train_index, test_index in skf.split(X, y):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	eval_set = [(X_test, y_test)]
	model = XGBClassifier(objective='f1_micro',max_depth=3,learning_rate=1.0)
	model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=10,verbose=True)
	y_pred = model.predict(X_test, ntree_limit=model.best_ntree_limit)
	scores.append(f1_score(y_test, y_pred,average='micro'))
	print '######################'
	print 'f1 score: ',scores[-1]
	print '######################'
#print scores





















