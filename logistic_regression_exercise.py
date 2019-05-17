import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ad_data=pd.read_csv('advertising.csv')

print(ad_data.head())
print(ad_data.columns)

ad_data['Age'].plot(kind='hist',bins=30)
plt.show()

sns.jointplot(x='Age',y='Area Income',data=ad_data)
plt.show()

sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,kind='kde',color='red')
plt.show()

sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data)
plt.show()

# sns.pairplot(ad_data,hue='Clicked on Ad')
# plt.show()

country=pd.get_dummies(ad_data['Country'],drop_first=True)
ad_data=pd.concat([ad_data,country],axis=1)
print(ad_data.columns)


from sklearn.model_selection import train_test_split
X = ad_data.drop(['Ad Topic Line','City','Country','Timestamp'],axis=1)
y = ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)

pred=logmodel.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,pred))




