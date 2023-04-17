# PCA is used to:
# 1. easily visualize multidimensional datasets.
# 2. pca is used to compress data.
# Eigenvectors are subtracted when making PCA

# We will reduce the size of the iris flower from 4 to 2.

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

url = 'pca_iris.data'
df = pd.read_csv(url, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])


features = ['sepal length', 'sepal width', 'petal length', 'petal width']

# Separate the features as x
x= df[features]

# Split the target as y
y = df[['target']]

# Standardizing the features
x = StandardScaler().fit_transform(x)

pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(x)
principalDF = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

principalDF

# Add target column to PCA dataframe
final_dataframe = pd.concat([principalDF, df[['target']]], axis = 1)
final_dataframe

# drawing basic graphic of final_dataframe
dfsetosa =final_dataframe[df.target == 'Iris-setosa']
dfvirginica = final_dataframe[df.target == 'Iris-virginica']
dfversicolor = final_dataframe[df.target == 'Iris-versicolor']
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.scatter(dfsetosa['principal component 1'], dfsetosa['principal component 2'], color = 'green')
plt.scatter(dfvirginica['principal component 1'], dfvirginica['principal component 2'], color = 'red')
plt.scatter(dfversicolor['principal component 1'], dfversicolor['principal component 2'], color = 'blue')

# Plotting more professional
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['g', 'b', 'r']

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

for target, col in zip(targets, colors):
    dftemp = final_dataframe[df.target == target]
    plt.scatter(dftemp['principal component 1'], dftemp['principal component 2'], color = col)

pca.explained_variance_ratio_


# PPCAercentage of data protected
pca.explained_variance_ratio_.sum()




