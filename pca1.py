import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer.keys())

df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
print(df.head())

#scaling is necessary
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(df)

scaled_data=scaler.transform(df)

#pca
from sklearn.decomposition import PCA
pca=PCA(n_components=2)

pca.fit(scaled_data)

x_pca=pca.transform(scaled_data)

print(scaled_data.shape)
print(x_pca.shape)

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='coolwarm')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal copmponent')
plt.show()
#Clearly by using these two components we can easily separate
# these two classes.

print(pca.components_)
#In this numpy matrix array, each row represents a principal component, and each
# column relates back to the original
# features. we can visualize this relationship with a heatmap:

df_comp=pd.DataFrame(pca.components_,columns=cancer['feature_names'])
plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma')
plt.show()