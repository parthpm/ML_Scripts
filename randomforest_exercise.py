import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

loans = pd.read_csv('loan_data.csv')

print(loans.head(),loans.info(),loans.describe())

#data analysyis
loans[loans['credit.policy']==1]['fico'].plot.hist(bins=30,label='1')
loans[loans['credit.policy']==0]['fico'].plot.hist(bins=30,color='red',label='0')
plt.legend()
plt.xlabel('Fico')
plt.show()

loans[loans['not.fully.paid']==1]['fico'].hist(bins=30,color='blue',label='1')
loans[loans['not.fully.paid']==0]['fico'].hist(bins=30,color='red',label='0')
plt.legend()
plt.xlabel('Fico')
plt.show()

sns.countplot(x='purpose',data=loans,hue='not.fully.paid')

plt.show()

plt.figure(figsize=(12,7))
sns.lmplot(col='not.fully.paid',x='fico',y='int.rate',data=loans,hue='credit.policy')
plt.show()

#removing categorical features

cat_feats=pd.get_dummies(loans['purpose'],drop_first=True,)
loans.drop(['purpose'],inplace=True,axis=1)
loans_final=pd.concat([loans,cat_feats],axis=1)

print(loans_final.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(loans_final.drop('not.fully.paid',axis=1),loans_final['not.fully.paid'],random_state=101,test_size=0.3)


from sklearn.tree import DecisionTreeClassifier
cl=DecisionTreeClassifier()
cl.fit(X_train,y_train)
#predictions

pred=cl.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))


#random forests

from sklearn.ensemble import RandomForestClassifier
rnn=RandomForestClassifier(criterion='entropy',n_estimators=100)
rnn.fit(X_train,y_train)
p=rnn.predict(X_test)

print(classification_report(y_test,p))
print(confusion_matrix(y_test,p))




