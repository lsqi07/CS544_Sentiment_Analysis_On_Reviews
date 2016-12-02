# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:26:44 2016

@author: Yue Shen
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import metrics
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Unnormalized Confusion matrix')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



#df = pd.read_csv(r'movie-pang02.csv')
#df = pd.read_csv(r'epinions3.csv')
df = pd.read_csv(r'twitter-sanders-apple.csv')
le = preprocessing.LabelEncoder()
df.label=le.fit_transform(df.label)

labels=df.label

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_df=0.5)
features= vectorizer.fit_transform(df.text)

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()#smooth_idf=False)
tfidf = transformer.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(tfidf,labels, test_size=0.25, random_state=50)

if 2 in y_test:
    class_names=['negative','neutral','positive']
else:
    class_names=['negative','positive']


start_time=time.time()                    
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=900)
clf = clf.fit(X_train, y_train)
y_pred=clf.predict(X_test) 
stop_time=time.time()
print("classifier run in %.2fs" % (stop_time - start_time)) 
print(np.mean(y_pred == y_test))
print(metrics.classification_report(y_test,y_pred))
RF_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(RF_matrix, classes=class_names,
                      title='Confusion matrix for RamdomForest')
           
            

start_time=time.time() 
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier( alpha=1e-5,activation='logistic',
                    hidden_layer_sizes=(10, 2),max_iter=1000)

clf.fit(X_train, y_train)                  
y_pred=clf.predict(X_test) 
stop_time=time.time()
print("classifier run in %.2fs" % (stop_time - start_time)) 
print(np.mean(y_pred == y_test))
MLP_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
print(metrics.classification_report(y_test,y_pred))
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(MLP_matrix, classes=class_names,
                      title='Confusion matrix for MLP')

                               
