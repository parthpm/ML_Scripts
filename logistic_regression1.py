import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#importing the data
train = pd.read_csv('titanic_train.csv')
print(train.head())

#Exploratory Data Analysis

#missing data
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='Dark2')
plt.show()

# sns.set_style(style='whitegrid')
#
# sns.countplot(x='Survived',hue='Sex',data=train)
# plt.show()
#
# sns.countplot(x='Survived',hue='Pclass',data=train)
# plt.show()
#
# sns.distplot(train['Age'].dropna(),kde=False,bins=30)
# plt.show()
#
# sns.countplot(x='SibSp',data=train)
# plt.show()

# sns.countplot(x='Fare',data=train)
# plt.show()

#data Cleaning
plt.figure(figsize=(12,8))
sns.boxplot(x='Pclass',y='Age',data=train)
plt.show()

def fun(cols):
    age=cols[0]
    pclass=cols[1]

    if pd.isnull(age):
        if pclass==1:
            return 37
        elif pclass==2:
            return 29
        else:
            return 24
    else:
        return age
train['Age']=train[['Age','Pclass']].apply(fun,axis=1)

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='Dark2')
plt.show()

train.drop('Cabin',axis=1,inplace=True)
train.dropna()

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='Dark2')
plt.show()


#converting the categorical features
sex=pd.get_dummies(train['Sex'],drop_first=True)
embarked=pd.get_dummies(train['Embarked'],drop_first=True)

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embarked],axis=1)

print(train.head())

#Building a Logistic Regression model
X=train.drop('Survived',axis=1)
y=train['Survived']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size=0.30,random_state=101)

#Training and Predicting
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X,y)

predictions=logmodel.predict(X_test)


#evaluations
from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))







