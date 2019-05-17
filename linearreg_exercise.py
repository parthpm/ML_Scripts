
import numpy as np

F,N=input().split()
F=int(F)
N=int(N)


A= [[float(j) for j in input().split()] for i in range(N)]

arr_2d=np.array(A)
X_train=arr_2d[:,:-1]
y_train=arr_2d[:,-1]


from sklearn.linear_model import LinearRegression
lm=LinearRegression()

lm.fit(X_train,y_train)

T=int(input())
B= [[float(j) for j in input().split()] for i in range(T)]
arr2_2d=np.array(B)

X_test=arr2_2d
pred=lm.predict(X_test)

for i in pred:
    print("%.2f" %(i))


