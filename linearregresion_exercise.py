import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

customers=pd.read_csv('Ecommerce Customers')

print(customers.head())
print(customers.describe())


sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)
plt.show()

# sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)
# plt.show()

sns.jointplot(x='Time on App',y='Length of Membership',data=customers,kind='hex')
plt.show()

# sns.pairplot(customers)
# plt.show()

sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)
plt.show()

X=customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y=customers['Yearly Amount Spent']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.linear_model import LinearRegression
lm=LinearRegression()

lm.fit(X_train,y_train)
print(pd.DataFrame(data=lm.coef_,index=X.columns,columns=['Coeff']))

predictions=lm.predict(X_test)

plt.scatter(x=y_test,y=predictions)
plt.show()

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

plt.hist(y,bins=50,color='red')
plt.show()

