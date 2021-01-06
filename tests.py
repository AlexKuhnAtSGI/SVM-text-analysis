import pandas as pd
import numpy as np
from nb import my_Naive_Bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re
import nltk

count = 0
def preprocess(data):
	global count
	count += 1
	
	plain_text = BeautifulSoup(data,"html.parser").get_text()
	letters = re.sub("[^a-zA-Z]", " ", plain_text).upper()
	tokens = nltk.word_tokenize(letters)
	stopwords = set(nltk.corpus.stopwords.words("english"))
	words = [w for w in tokens if not w in stopwords]
	words = [nltk.stem.SnowballStemmer("english").stem(w) for w in words]
	cleaned = " ".join(words)
	if (count % 5000 == 0):
		print("cleaned!", count)
		
	return cleaned

df = pd.read_csv('train.csv')
df['sentiment'] = np.where(df['sentiment'].str.contains('positive'), 1, 0)
y_train = np.array(df['sentiment'])

df['clean'] = df['review'].apply(preprocess)

#bernoulli NB classifiers require that features be binary, an indicator of whether the word was present or absent, so the training data will differ for my classifier - SVM performs better on this as well
vectorizer = CountVectorizer(max_features=5000, binary=True)
X_train = vectorizer.fit_transform(df['clean']).toarray()
X_train = np.array(X_train)
print(vectorizer.get_feature_names())

#here i create an alternate version of the training data, retaining frequencies of words, for the multinomial classifier
vectorizer_alt = CountVectorizer(max_features=5000)
X_train_alt = vectorizer_alt.fit_transform(df['clean']).toarray()
X_train_alt = np.array(X_train_alt)

def svc_opt():
	kf = KFold(n_splits=5, shuffle=True)
	scores_svm = []
	Cvals = [0.001, 0.005, 0.008, 0.01, 0.05, 0.1, 0.5, 1, 5]
	for c in Cvals:
		scores = []
		clf_svm = LinearSVC(C=c, dual = False)
		for train_index, test_index in kf.split(X_train):
			X_tr, X_val = X_train[train_index], X_train[test_index]
			y_tr, y_val = y_train[train_index], y_train[test_index]
			clf_svm.fit(X_tr, y_tr)
			
			scores.append(clf_svm.score(X_val, y_val))
		scores_svm.append(np.mean(scores))
		print("score of", np.mean(scores), "obtained at c of", c)
		
	plt.plot(Cvals, scores_svm, 'rx')
	plt.plot(Cvals, scores_svm)
	plt.title("Linear SVM")
	plt.xscale('log')
	plt.xlabel("Value of C")
	plt.ylabel("Validation accuracy")
	plt.show()
	
def multi_opt():
	kf = KFold(n_splits=5, shuffle=True)
	scores_multi = []
	alphas = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 8, 10]
	for a in alphas:
		scores = []
		clf_multi = MultinomialNB(alpha=a)
		for train_index, test_index in kf.split(X_train_alt):
			X_tr, X_val = X_train_alt[train_index], X_train_alt[test_index]
			y_tr, y_val = y_train[train_index], y_train[test_index]
			clf_multi.fit(X_tr, y_tr)
			
			scores.append(clf_multi.score(X_val, y_val))
		scores_multi.append(np.mean(scores))
		print("score of", np.mean(scores), "obtained at alpha of", a)
		
	plt.plot(alphas, scores_multi, 'rx')
	plt.plot(alphas, scores_multi)
	plt.title("Multinomial NB")
	#plt.xscale('log')
	plt.xlabel("Value of Alpha")
	plt.ylabel("Validation accuracy")
	plt.ylim(0.81,0.86)
	plt.show()
	
def kfold_opt():
	folds = np.array([5,10])
	clf_bernoulli = my_Naive_Bayes()
	clf_multi = MultinomialNB()
	clf_svm = LinearSVC(C=0.005, dual=False)
	loops = 5
	
	avg_scores_5 = [0,0,0]
	avg_scores_10 = [0,0,0]
	
	for i in range (loops):
		for fold in folds:
			kf = KFold(n_splits=fold, shuffle=True)
			for train_index, test_index in kf.split(X_train_alt):
				X_tr, X_val = X_train[train_index], X_train[test_index]
				y_tr, y_val = y_train[train_index], y_train[test_index]
				X_tr_alt, X_val_alt = X_train_alt[train_index], X_train_alt[test_index]
				
				clf_bernoulli.fit(X_tr, y_tr)
				clf_multi.fit(X_tr_alt, y_tr)
				clf_svm.fit(X_tr, y_tr)
				
				y_pred = clf_bernoulli.predict(X_val)
				score_bernoulli= clf_bernoulli.score(y_pred, y_val)
				score_multi = clf_multi.score(X_val_alt, y_val)
				score_svm = clf_svm.score(X_val, y_val)
				#print(score_svm)
				
				if (fold==5):
					avg_scores_5[0] += score_bernoulli
					avg_scores_5[1] += score_multi
					avg_scores_5[2] += score_svm
				else:
					avg_scores_10[0] += score_bernoulli
					avg_scores_10[1] += score_multi
					avg_scores_10[2] += score_svm
		print ("iteration", i, "complete")
	
	avg_scores_5 = np.array(avg_scores_5)/(loops*folds[0])
	avg_scores_10 = np.array(avg_scores_10)/(loops*folds[1])
	
	bernoulli_scores = [avg_scores_5[0],avg_scores_10[0]]
	multi_scores = [avg_scores_5[1],avg_scores_10[1]]
	svm_scores = [avg_scores_5[2],avg_scores_10[2]]
	
	ax = plt.subplot(111)
	plt.bar(x=(folds-0.5), height=bernoulli_scores, label='Bernoulli NB', width=0.5)
	plt.bar(x=folds, height=multi_scores, label='Multinomial NB', width=0.5)
	plt.bar(x=(folds+0.5), height=svm_scores, label='Linear SVC', width=0.5)
	plt.xlabel('# Folds')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.show()
			
		
#svc_opt()
#multi_opt()
#kfold_opt()

clf = LinearSVC(C=0.005, dual=False)
kf = KFold(n_splits=5, shuffle=True)
bestScore = 0

for i in range(5):
	for train_index, test_index in kf.split(X_train):
		X_tr, X_val = X_train[train_index], X_train[test_index]
		y_tr, y_val = y_train[train_index], y_train[test_index]

		clf.fit(X_tr, y_tr)
		
		currScore = clf.score(X_val, y_val)
		
		if (currScore > bestScore):
			best = clf
			bestScore = currScore
		print("done iteration")
	print("Training complete")
	
print (bestScore)
clf = best

df_test = pd.read_csv('test.csv')
df_test['clean'] = df_test['review'].apply(preprocess)
vectorizer = CountVectorizer(max_features=5000, binary=True)
X_test = vectorizer.fit_transform(df_test['clean']).toarray()

X_test = np.array(X_test)
y_pred = clf.predict(X_test)

y_pred = y_pred.astype('str')
y_pred = np.where(y_pred == '1', 'positive', 'negative')
df_test['sentiment'] = y_pred

df_test.index.name='id'
df_test['sentiment'].to_csv('predictions.csv', index=True, header=True)