import nltk
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import string

messages=pd.read_csv('SMSSpamCollection',sep='\t',names=['label','message'])

messages['length']=messages['message'].apply(len)
print(messages.head())

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of cleaned text
    """
    nopun=[char for char in mess if char not in string.punctuation]
    nopun=''.join(nopun)

    clean_mess=[word for word in nopun.split() if word.lower() not in stopwords.words('english')]

    return clean_mess

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

#train test Split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = \
    train_test_split(messages['message'], messages['label'], test_size=0.2)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))

from sklearn.pipeline import Pipeline
pipline=Pipeline([('bow',CountVectorizer(analyzer=text_process)),('tfidf', TfidfTransformer()),
                  ('classifier', MultinomialNB())])

pipline.fit(msg_train,label_train)

predictions=pipline.predict(msg_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(predictions,label_test))
print(confusion_matrix(predictions,label_test))




