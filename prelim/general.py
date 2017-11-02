import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity
from scipy.stats import skew

#load data
xtrain = np.genfromtxt('../../contest_data/train.csv', delimiter=',')[1:,1:-1]
ytrain = np.genfromtxt('../../contest_data/train.csv', delimiter=',')[1:,-1]
ytrain=np.asmatrix(ytrain).T

#plotting number of missing values in each column
missings=np.sum(np.isnan(xtrain),axis=0)
plt.plot(missings)
plt.xlabel('column index')
plt.ylabel('number of missing values')
plt.title('missing values vs column')
plt.savefig('missings.png')

#plotting proprtion of missing values in each class
prop=[]
for i in range(29):
	ind=np.ravel(ytrain==float(i))
	missings=float(np.sum(np.isnan(xtrain[ind,:])))
	nonmissings=float(np.sum(np.isfinite(xtrain[ind,:])))
	pi=missings/(missings+nonmissings)
	print pi
	prop.append(pi)

plt.plot(prop)
plt.xlabel('column index')
plt.ylabel('percentage of missing values')
plt.title('percentage missing values vs class')
#plt.show()
plt.savefig('missings_classwise.png')

#unconditional mean, variance and skew of all cols
means=np.array([np.mean(xtrain[np.isfinite(xtrain[:,i]),i]) for i in range(2600)])
var=np.array([np.var(xtrain[np.isfinite(xtrain[:,i]),i]) for i in range(2600)])
skews=np.array([skew(xtrain[np.isfinite(xtrain[:,i]),i]) for i in range(2600)])

#class conditioned mean, variance and skew of all cols

	means=np.array([np.mean(xtrain[np.isfinite(xtrain[:,i]),i]) for i in range(2600)])
	var=np.array([np.var(xtrain[np.isfinite(xtrain[:,i]),i]) for i in range(2600)])
	skews=np.array([skew(xtrain[np.isfinite(xtrain[:,i]),i]) for i in range(2600)])

#histogram of column 1
finite=np.isfinite(xtrain[:,0])
plt.hist(ytrain[finite,0],bins=29)
plt.show()

#unconditional density estimation of column 1 
X=xtrain[finite,0][:,np.newaxis]
X_plot=np.linspace(0,1,1000)[:,np.newaxis]
kde = KernelDensity(kernel='gaussian', bandwidth=0.07).fit(X)
log_dens = kde.score_samples(X_plot)
dens=np.exp(log_dens)
plt.hist(X,bins=100,normed=True)
plt.plot(X_plot,dens)
plt.show()


#proportion of missing data in classes of col 1
for i in range(29):
	ind=ytrain==i
	missings=sum(np.isnan(xtrain[ind,:500]))
	nonmissings=sum(np.isfinite(xtrain[ind,:500]))
	print missings/nonmissings








