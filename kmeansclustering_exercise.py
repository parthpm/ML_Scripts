import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('College_Data',index_col=0)
print(data.head())

sns.set_style('whitegrid')
sns.lmplot('Room.Board','Grad.Rate',data=data, hue='Private',fit_reg=False)
plt.show()

sns.lmplot(y='F.Undergrad',x='Outstate',data=data, hue='Private',fit_reg=False)
plt.show()

data[data['Private']=='Yes']['Outstate'].hist(color='blue',bins=33)
data[data['Private']=='No']['Outstate'].hist(color='red',bins=33)
plt.show()

data[data['Private']=='Yes']['Grad.Rate'].hist(color='blue',bins=33)
data[data['Private']=='No']['Grad.Rate'].hist(color='red',bins=33)

g=sns.FacetGrid(data=data,hue='Private',aspect=2,size=6)
g=g.map(plt.hist,'Outstate',bins=33,alpha=0.7)
plt.show()

from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=2)

kmeans.fit(data.drop('Private',axis=1))

print(kmeans.cluster_centers_)
print(kmeans.labels_)

cluster=pd.get_dummies(data['Private'],drop_first=True)
print(cluster)

data_final=pd.concat([data,cluster],axis=1)
print(data_final.head())

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(cluster,kmeans.labels_))
print(classification_report(cluster,kmeans.labels_))



