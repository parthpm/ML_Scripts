import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('Classified Data',index_col=0)
print(df.head())   #Set index_col=0 to use the first column as the index

#Standardize the Variables
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))

scaled_features=scaler.transform(df.drop('TARGET CLASS',axis=1))

df_feat=pd.DataFrame(data=scaled_features,columns=df.columns[:-1])
print(df_feat.head())

#Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(scaled_features,df['TARGET CLASS'],test_size=0.3,random_state=101)

#using knn
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

#predictions

predictions=knn.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


#Chosing a K value
error_rate=[]

for i in range(1,40):

    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i!=y_test))

plt.plot(range(1,40),error_rate,marker='o')
plt.xlabel('K Values ')
plt.ylabel('Error')
plt.title('Eroor Graph')
plt.show()

#with k=23

knn=KNeighborsClassifier(n_neighbors=23)

knn.fit(X_train,y_train)

#predictions

predictions=knn.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))




