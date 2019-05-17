import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('KNN_Project_Data')
print(df.head())

# sns.pairplot(df,hue='TARGET CLASS')
# plt.show()

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_data=scaler.transform(df.drop('TARGET CLASS',axis=1))
df_feat=pd.DataFrame(scaled_data,columns=df.columns[:-1])

print(df_feat.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(df_feat,df['TARGET CLASS'],test_size=0.3,random_state=101)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

pred=knn.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

k_error=[]

for i in range(1,40):

    knn_i=KNeighborsClassifier(n_neighbors=i)
    knn_i.fit(X_train,y_train)
    pred=knn_i.predict(X_test)
    k_error.append(np.average(pred!=y_test))

plt.plot(range(1,40),k_error,marker='o',color='blue'x)
plt.show()

knn_i=KNeighborsClassifier(n_neighbors=31)
knn_i.fit(X_train,y_train)
preds=knn_i.predict(X_test)
print(confusion_matrix(y_test,preds))
print(classification_report(y_test,preds))

