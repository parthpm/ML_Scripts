import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('kyphosis.csv')

print(df.head())




from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(df.drop('Kyphosis',axis=1))
scaled_data=scaler.transform(df.drop('Kyphosis',axis=1))
df_feat=pd.DataFrame(data=scaled_data,columns=df.columns[1:])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(df_feat,df['Kyphosis'])

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)

predi=knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,predi))
print(classification_report(y_test,predi))
