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
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout

#X = np.genfromtxt('../../contest_data/xtrain_linear_imputed.csv', delimiter=',')
X=np.load('../../contest_data/xtrain_linear_imputed.npy')
print 'loaded X'
#y = np.genfromtxt('../../contest_data/train.csv', delimiter=',')[1:,-1]
y=np.load('../../contest_data/ytrain.npy')
print 'loaded y'

pca = PCA(n_components=100)
X_pca = pca.fit(X).transform(X)

enc = OneHotEncoder()
enc.fit(y[:,np.newaxis]) 
y_hot=enc.transform(y[:,np.newaxis]).toarray()

model = Sequential([
    Dense(500, input_shape=(2600,)),
    Activation('relu'),
    #Dropout(0.2),
    Dense(100),
    Activation('relu'),
    #Dropout(0.2),
    Dense(100),
    Activation('relu'),
    Dense(29),
    Activation('softmax'),
])

model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='auto')]

model.fit(X, y_hot, epochs=1000, batch_size=16,validation_split=0.1,callbacks=callbacks)

#neural nets do bette with PCA