import nltk
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



messages = [line.rstrip() for line in open('SMSSpamCollection')]
print(len(messages))
# print(messages)

print(messages[50])

for no,message in enumerate(messages[:5]):
    print(no,message)
    print("\n")

messages=pd.read_csv('SMSSpamCollection',sep='\t',names=['label','message'])
print(messages.head())

#data analysyis

print(messages.describe())
print(messages.groupby('label').describe())

messages['length']=messages['message'].apply(len)
print(messages.head())

# messages['length'].plot.hist(bins=150)
# plt.show()
# print(messages.length.describe())

print(messages[messages['length']==910]['message'].iloc[0])

# messages.hist(column='length', by='label', bins=50,figsize=(12,4))
# plt.show()

# sns.FacetGrid(col='label',data=messages,).map(plt.hist(bins=150,figsize=(12,4)),'length')
# plt.show()
#text Preprocessing

import string

mess = 'Sample message! Notice: it has punctuation.'

# Check characters to see if they are in punctuation
nopunc = [char for char in mess if char not in string.punctuation]

# Join the characters again to form the string.
nopunc = ''.join(nopunc)
print(nopunc)

print('Some stopwords are:',stopwords.words('english')[:10])
print(nopunc.split())
clean_mess=[word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
print(clean_mess)

#creating a function to do clean up processs

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

print(messages.head())

print(messages['message'].head(5).apply(text_process))

#vectoization
#Each vector will have as many dimensions as there are unique words
# in the SMS corpus. We will first use SciKit Learn's CountVectorizer.
#  This model will convert
#  a collection of text documents to a matrix of token counts.

from sklearn.feature_extraction.text import CountVectorizer

# Might take awhile...
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

# Print total number of vocab words
print(len(bow_transformer.vocabulary_))

#taking one message
message4=messages['message'][3]
print(message4)

bow4=bow_transformer.transform([message4])
print(bow4)
print(bow4.shape)

print(bow_transformer.get_feature_names()[4068])
print(bow_transformer.get_feature_names()[9554])

#taking entire message column

messages_bow=bow_transformer.transform(messages['message'])

print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)

sparsity=messages_bow.nnz/(messages_bow.shape[0]*messages_bow.shape[1])
print('Sparsity is {}'.format(sparsity))

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer=TfidfTransformer()
tfidf_transformer.fit(messages_bow)

#for one observation

tfidf4=tfidf_transformer.transform(bow4)
print(tfidf4)

#check idf for u and university word
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])

messages_tfidf=tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)

#Training a Model

from sklearn.naive_bayes import MultinomialNB
spam_detect_model=MultinomialNB().fit(messages_tfidf,messages['label'])

#predictions
all_pred=spam_detect_model.predict(messages_tfidf)
from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(messages['label'],all_pred))
print(classification_report(messages['label'],all_pred))










