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
import tflearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import tensorflow as tf 
from sklearn.utils import resample

X = np.genfromtxt('../../contest_data/xtrain_linear_imputed.csv', delimiter=',')
print 'loaded X'
y = np.genfromtxt('../../contest_data/train.csv', delimiter=',')[1:,-1]
print 'loaded y'

enc = OneHotEncoder()
enc.fit(y[:,np.newaxis]) 
y_hot=enc.transform(y[:,np.newaxis]).toarray()

#always reset before starting
tf.reset_default_graph()

# Build neural network
net = tflearn.input_data(shape=[None, 2600])
net = tflearn.fully_connected(net, 200)
#net = tflearn.layers.core.dropout(net, 0.95)
net = tflearn.fully_connected(net, 50)
#net = tflearn.layers.core.dropout(net, 0.95)
#net = tflearn.fully_connected(net, 80)
#net = tflearn.layers.core.dropout(net, 0.95)
net = tflearn.fully_connected(net, 29, activation='softmax')
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)


best_score=0
skf = StratifiedKFold(n_splits=10)
logs4=[]
for train_index, test_index in skf.split(X_norm, y):
	X_train, X_test = X_norm[train_index], X_norm[test_index]
	y_train, y_test = y_hot[train_index,:], y_hot[test_index,:]
	for i in range(1,20):	
		model.fit(X_train, y_train, n_epoch=1, batch_size=16, show_metric=True)
		y_pred = model.predict(X_test)
		score=f1_score(np.argmax(y_test,axis=1), np.argmax(y_pred,axis=1),average='micro')
		print '######################'
		print 'f1 score: ',score
		print '######################'
		logs4.append(score)
		if score>best_score:
			best_score=score
			best_i=i
			#model.save("best_model.tfl")
	#model.load('best_model.tfl')
	print "best score: ",best_score
	break


# Start training (apply gradient descent algorithm)
skf = StratifiedKFold(n_splits=10)
for train_index, test_index in skf.split(X, y):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y_hot[train_index,:], y_hot[test_index,:]
	model.fit(X_train, y_train, n_epoch=40, batch_size=16, show_metric=True,validation_set=0.1)
	y_pred = model.predict(X_test)
	score=f1_score(np.argmax(y_test,axis=1), np.argmax(y_pred,axis=1),average='micro')
	print '######################'
	print 'f1 score: ',score
	print '######################'
	break



plt.plot(range(19),logs[:19])
plt.plot(range(19),logs2)
plt.plot(range(19),logs3)
plt.plot(range(19),logs4)
plt.legend(('1500,500,500','1000,300,100','500,250,80','200,50'))
plt.xlabel('epochs')
plt.ylabel('f1 score')
plt.title('neural network performance over epochs')
plt.savefig('NN.png')
#plt.show()





#bagging

skf = StratifiedKFold(n_splits=10)
for train_index, test_index in skf.split(X, y):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y_hot[train_index,:], y_hot[test_index,:]
	
	mods=[]
	try:del(classes)
	except NameError: pass
	for i in range(10):
		#always reset before starting
		tf.reset_default_graph()

		# Build neural network
		net = tflearn.input_data(shape=[None, 2600])
		net = tflearn.fully_connected(net, 500)
		#net = tflearn.layers.core.dropout(net, 0.95)
		net = tflearn.fully_connected(net, 100)
		#net = tflearn.layers.core.dropout(net, 0.95)
		net = tflearn.fully_connected(net, 50)
		#net = tflearn.layers.core.dropout(net, 0.95)
		net = tflearn.fully_connected(net, 29, activation='softmax')
		net = tflearn.regression(net)

		# Define model
		model = tflearn.DNN(net)

		# bootstrapping
		X_boot, y_boot = resample(X_train, y_train)
		
		#fit
		model.fit(X_boot, y_boot, n_epoch=18, batch_size=16, show_metric=True)
		
		y_pred=model.predict(X_test)
		score=f1_score(np.argmax(y_test,axis=1), np.argmax(y_pred,axis=1),average='micro')
		
		print '######################'
		print 'model number:',i
		print 'f1 score: ',score
		print '######################'

		mods.append(model)
	preds=np.zeros(y_test.shape)
	for mod in mods:
		y_pred=mod.predict(X_test)
		preds=preds+y_pred
	score=f1_score(np.argmax(y_test,axis=1), np.argmax(preds,axis=1),average='micro')	
	break

#27%


1500 500 500
1000 300 100
500 250 80
200 50












