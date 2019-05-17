import seaborn as sns
import matplotlib.pyplot as plt

#create some data
from sklearn.datasets import make_blobs

data=make_blobs(n_samples=200,n_features=2,centers=4,cluster_std=1.8,random_state=101)

print(data)
print(type(data))

#visualize the data
plt.scatter(x=data[0][:,0],y=data[0][:,1],c=data[1],cmap='rainbow')
plt.show()

#knn means cluster

from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=4)

kmeans.fit(data[0])
print(kmeans.cluster_centers_)

print(kmeans.labels_)

#subplot
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(12,8),sharey=True)

axes[0].set_title('Kmmeans')
axes[1].set_title('original')

axes[0].scatter(x=data[0][:,0],y=data[0][:,1],c=kmeans.labels_,cmap='rainbow')
axes[1].scatter(x=data[0][:,0],y=data[0][:,1],c=data[1],cmap='rainbow')
plt.show()

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(data[1],kmeans.labels_))
print(confusion_matrix(data[1],kmeans.labels_))
