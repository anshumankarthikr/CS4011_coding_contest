import numpy as np
import matplotlib.pyplot as plt

#load data
xtrain = np.genfromtxt('../../contest_data/train.csv', delimiter=',')[1:,1:-1]
ytrain = np.genfromtxt('../../contest_data/train.csv', delimiter=',')[1:,-1]
ytrain=np.asmatrix(ytrain).T

for i  in range(500):
	 
