# motivated by ensemble of deep lstm learners
import numpy as np
import scipy.io
import pandas as pd

def loadingDB(fileDir, DB=79):
	
	if DB==79: 
		matfile = fileDir+'opp.mat'
		print(matfile)
		data = scipy.io.loadmat(matfile)
		
		X_train = np.transpose(data['trainingData'])
		X_valid = np.transpose(data['valData'])
		X_test = np.transpose(data['testingData'])
		print('normalising... zero mean, unit variance')
		mn_trn = np.mean(X_train, axis=0)
		std_trn = np.std(X_train, axis=0)
		X_train = (X_train - mn_trn)/std_trn
		X_valid = (X_valid - mn_trn)/std_trn
		X_test = (X_test - mn_trn)/std_trn
		print('normalising...X_train, X_valid, X_test... done')
		y_train = data['trainingLabels'].reshape(-1)-1
		y_valid = data['valLabels'].reshape(-1)-1
		y_test = data['testingLabels'].reshape(-1)-1
		print('loading the 79-dim matData successfully . . .')

	if DB==60:
		matfile = fileDir+'skoda.mat'
		data = scipy.io.loadmat(matfile)

		X_train = data['X_train']
		X_valid = data['X_valid']
		X_test = data['X_test']
		y_train = data['y_train'].reshape(-1)
		y_valid = data['y_valid'].reshape(-1)
		y_test = data['y_test'].reshape(-1)
		print('the Skoda dataset was normalized to zero-mean, unit variance')
		print('loading the 33HZ 60d matData successfully . . .')

	if DB==9:
		matfile = fileDir+'FOG.mat'
		data = scipy.io.loadmat(matfile)
		
		X_train = data['X_train']
		X_valid = data['X_valid']
		X_test = data['X_test']
		y_train = data['y_train'].reshape(-1)
		y_valid = data['y_valid'].reshape(-1)
		y_test = data['y_test'].reshape(-1)
		print('binary classification problem . . . ')
		print('the FOG dataset was normalized to zero-mean, unit variance')
		print('loading the 32HZ FOG 9d matData successfully . . .')
	
	if DB==52:
		matfile = fileDir+'pamap2.mat'
		data = scipy.io.loadmat(matfile)
		
		X_train = data['X_train']
		X_valid = data['X_valid']
		X_test = data['X_test']
		y_train = data['y_train'].reshape(-1)
		y_valid = data['y_valid'].reshape(-1)
		y_test = data['y_test'].reshape(-1)
		print('the PAMAP2 dataset was normalized to zero-mean, unit variance')
		print('loading the 33HZ PAMAP2 52d matData successfully . . .')
	
	X_train = X_train.astype(np.float32)
	X_valid = X_valid.astype(np.float32)
	X_test = X_test.astype(np.float32)
   
	y_train = y_train.astype(np.uint8)
	y_valid = y_valid.astype(np.uint8)
	y_test = y_test.astype(np.uint8)
	
	return X_train, X_valid, X_test, y_train, y_valid, y_test
