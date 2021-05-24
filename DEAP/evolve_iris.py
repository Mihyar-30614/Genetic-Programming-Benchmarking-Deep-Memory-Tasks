# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

# importing the modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.metrics import silhouette_score

# Loading the Data Set and Displaying first 5 rows
iris = datasets.load_iris()
x = iris.data
y = iris.target

data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target_names'])

print(data1.head())

# Visualizing relationship between different atrributes in the Data
sns.pairplot(data1, hue='target_names', height=2.5)
plt.show()

# Elbow Finding
sns.set()
yval = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    yval.append(kmeans.inertia_)
    
#Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 11), yval)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Score') #within cluster sum of squares
plt.show()

# Ideal cluster number should be = 2
silh = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    cluster_labels = kmeans.fit_predict(x)
    s = silhouette_score(x, labels = cluster_labels)
    silh.append(s)
    
plt.plot(range(2, 11), silh)
plt.title('Silhoutte Index Values')
plt.xlabel('Number of clusters')
plt.ylabel('Score') #within cluster sum of squares
plt.show()