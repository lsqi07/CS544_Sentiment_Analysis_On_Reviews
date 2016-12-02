from pandas import DataFrame
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import re

# read the input csv file
df = pd.read_csv(r'twitter-sanders-apple.csv')
le = preprocessing.LabelEncoder()
df.label=le.fit_transform(df.label)
word_dict = {}

# a set of all stop words
stopwords = set(["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with"]) 

# get word list for each line in csv file
def get_words(item):
    lower_words = re.split(r'[^a-zA-Z]', item.lower())
    word_list = ' '.join(lower_words).split()
    return word_list

# get a dictionary which includes each word and its frequency in whole data set
for item in df.text:
    words = get_words(item)
    for word in words:
        if word in stopwords : continue
        word_dict[word] = word_dict.get(word, 0)+1

word_idx = {}
idx = 0

# get a dictionary which includes each word and the corresponding index
for word in word_dict:
    word_idx[word] = idx
    idx += 1

# get features of all line in the input csv file
def get_features(item):
    rs = []
    for i in range(len(word_idx)):
        rs.append(0)
    words = get_words(item)
    for word in words:
        if word in stopwords : continue
        rs[word_idx[word]] += 1
    return rs

# get features as X_train and labels as y_train
features=[]
for item in df.text:
    features.append(get_features(item))
labels=df.label


# create features using toolkit

# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer(min_df=1)
# features= vectorizer.fit_transform(df.text)

# from sklearn.feature_extraction.text import TfidfTransformer
# transformer = TfidfTransformer(smooth_idf=False)
# tfidf = transformer.fit_transform(features)

# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(min_df=1)
# tfidf=vectorizer.fit_transform(df.text)




X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size=0.15, random_state=50)

# Naive Bayes
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB() 
bnb.fit(X_train, y_train) 
y_pred=bnb.predict(X_test) 
print(np.mean(y_pred == y_test)) 
print(metrics.classification_report(y_test,y_pred))
print(confusion_matrix(y_test, y_pred))

# SGD
from sklearn.linear_model import SGDClassifier 
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(X_train, y_train) 
y_pred=clf.predict(X_test) 
print(np.mean(y_pred == y_test))
print(metrics.classification_report(y_test,y_pred))
print(confusion_matrix(y_test, y_pred))

