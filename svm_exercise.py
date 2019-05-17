import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

iris=sns.load_dataset('iris')
print(iris.head())

sns.pairplot(iris,palette='Dark2',hue='species')
plt.show()

sns.jointplot(x='sepal_width',y='sepal_length',kind='kde',data=iris[iris['species']=='setosa'])
plt.show()

X=iris.drop('species',axis=1)
y=iris['species']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=101,test_size=0.3)

from sklearn.svm import SVC
classify=SVC(C=0.1)
classify.fit(X_train,y_train,)

pred=classify.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))

from sklearn.model_selection import GridSearchCV
param_grid={'C':[0.1,1, 10, 100, 1000],'gamma':[1,0.1,0.01,0.001,0.0001]}

gcv=GridSearchCV(SVC(),param_grid)
gcv.fit(X_train,y_train)

grid_pred=gcv.predict(X_test)

print(confusion_matrix(y_test,grid_pred))
print(classification_report(y_test,grid_pred))


