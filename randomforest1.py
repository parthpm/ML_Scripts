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
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3)

#decision Tree
from sklearn.tree import  DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)

prdictions=dtree.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,prdictions))
print(classification_report(y_test,prdictions))

#random forests

from sklearn.ensemble import RandomForestClassifier
rnf=RandomForestClassifier(n_estimators=10)
rnf.fit(X_train,y_train)

pr=rnf.predict(X_test)

print(confusion_matrix(y_test,pr))
print(classification_report(y_test,pr))


