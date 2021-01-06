import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold





class my_Naive_Bayes:
	def fit(self, X, y):
		num_samples = X.shape[0]
		separated = [[x for x, a in zip(X, y) if a == b] for b in np.unique(y)]
		
		numer = np.array([np.array(i).sum(axis=0) for i in separated]) + 1
		denom = np.array([len(i) + 2 for i in separated])
		
		self.probs = numer / denom[np.newaxis].T

	def predict_log_proba(self, X):
		deltas = []
		
		for x in X:
			deltas.append((np.log(self.probs) * x + \
				 np.log(1 - self.probs) * np.abs(x - 1)
				).sum(axis=1))
				
		return deltas

	def predict(self, X):
		pred = np.argmax(self.predict_log_proba(X), axis=1)
		return pred
		
	def score(self, y_pred, y):
		count = 0
		
		for i in range(len(y)):
			if y_pred[i] == y[i]:
				count += 1
		
		return count/len(y)