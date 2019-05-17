import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


USAhousing = pd.read_csv('USA_Housing.csv')
print(USAhousing.describe().head(),end='\n\n\n')
print(USAhousing.columns,end='\n\n\n')

#sns.pairplot(USAhousing)
#plt.show()

sns.distplot(USAhousing['Price'])
plt.show()

sns.heatmap(USAhousing.corr(),cmap='coolwarm')
plt.show()

X=USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']]
y=USAhousing['Price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.04,random_state=101)

from sklearn.linear_model import LinearRegression
lm=LinearRegression()

lm.fit(X_train,y_train)

print(lm.intercept_)

coef_df=pd.DataFrame(data=lm.coef_,index=X.columns,columns=['Coeffecient'])
print(coef_df)

predictions=lm.predict(X_test)

sns.jointplot(kind='scatter',y=y_test,x=predictions)
plt.show()
sns.distplot((y_test-predictions),bins=60)
plt.show()
p=lm.predict(np.array([79545.458574,5.682861,7.009188,4.09,23086.800503]).reshape(1,-1))
print(p)


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))