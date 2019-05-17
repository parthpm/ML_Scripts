import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('kyphosis.csv')

print(df.head())

sns.pairplot(df,hue='Kyphosis')
plt.show()

from  sklearn.model_selection import train_test_split
X=df.drop('Kyphosis',axis=1)
y=df['Kyphosis']
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.4,random_state=101)

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)
pred=classifier.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
# print(pd.DataFrame(y_test,pred,columns=['Test','Predictions']))
print(pred)
print(type(pred))
print(y_test,type(y_test))


