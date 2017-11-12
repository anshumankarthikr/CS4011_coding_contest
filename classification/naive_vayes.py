from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.ensemble import ExtraTreesClassifier
from mlxtend.feature_extraction import PrincipalComponentAnalysis as PCA
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

X = np.genfromtxt('../../contest_data/xtrain_linear_imputed.csv', delimiter=',')
y = np.genfromtxt('../../contest_data/train.csv', delimiter=',')[1:,-1]



gnb = GaussianNB()
scores = cross_val_score(gnb, X_norm, y,scoring='f1_micro',cv=5,verbose=5)
print scores.mean()

a=[]
for i in range(100,2700,100):
	pca = PCA(n_components=i)
	X_pca = pca.fit(X_norm).transform(X_norm)
	gnb = GaussianNB()
	scores = cross_val_score(gnb, X_pca, y,scoring='f1_micro',cv=5,verbose=5,n_jobs=-1)
	print scores.mean()
	a.append(scores.mean())
plt.plot(range(100,2700,100),a)
plt.xlabel('PCA dimensions')
plt.ylabel('f1')
plt.title('PCA dimensions vs naive bayes f1 score')
plt.savefig('bayes_pca.png')
