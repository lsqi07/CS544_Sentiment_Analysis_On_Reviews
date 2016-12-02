# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 10:52:35 2016

@author: Yue Shen
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv(r'twitter-sanders-apple.csv')
le = preprocessing.LabelEncoder()
df.label=le.fit_transform(df.label)

labels=df.label

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_df=0.5)
features= vectorizer.fit_transform(df.text)

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()#smooth_idf=False
tfidf = transformer.fit_transform(features)

scores = list()
scores_std = list()                    
from sklearn.model_selection import cross_val_score
n_estimators=[10, 50, 200, 300, 400,500,700,900]
for N in n_estimators:
 clf = RandomForestClassifier(n_estimators=N)
 this_scores = cross_val_score(
       clf, tfidf, labels, cv=4, scoring='f1_macro')
 scores.append(np.mean(this_scores))
 scores_std.append(np.std(this_scores))


scores, scores_std = np.array(scores), np.array(scores_std)

plt.figure().set_size_inches(8, 6)
plt.semilogx(n_estimators, scores)

# plot error lines showing +/- std. errors of the scores
std_error = scores_std / np.sqrt(4)

plt.semilogx(n_estimators, scores + std_error, 'b--')
plt.semilogx(n_estimators, scores - std_error, 'b--')

# alpha=0.2 controls the translucency of the fill color

plt.ylabel('CV score +/- std error')
plt.xlabel('n_estimators')
plt.axhline(np.max(scores), linestyle='--', color='.5')

 