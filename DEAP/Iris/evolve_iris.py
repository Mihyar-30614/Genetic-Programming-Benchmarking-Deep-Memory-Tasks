# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

# importing the modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Loading the Data Set and Displaying first 5 rows
iris = datasets.load_iris()
x = iris.data
y = iris.target

dataset = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                       columns=iris['feature_names'] + ['target_names'])


def plot():
    print(dataset.head())

    # Visualizing relationship between different atrributes in the Data
    sns.pairplot(dataset, hue='target_names', height=2.5)
    plt.show()

    # Elbow Finding
    sns.set()
    yval = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++',
                        max_iter=300, n_init=10, random_state=0)
        kmeans.fit(x)
        yval.append(kmeans.inertia_)

    # Plotting the results onto a line graph, allowing us to observe 'The elbow'
    plt.plot(range(1, 11), yval)
    plt.title('The elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Score')  # within cluster sum of squares
    plt.show()

    # Ideal cluster number should be = 2
    silh = []
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++',
                        max_iter=300, n_init=10, random_state=0)
        kmeans.fit(x)
        cluster_labels = kmeans.fit_predict(x)
        s = silhouette_score(x, labels=cluster_labels)
        silh.append(s)

    plt.plot(range(2, 11), silh)
    plt.title('Silhoutte Index Values')
    plt.xlabel('Number of clusters')
    plt.ylabel('Score')  # within cluster sum of squares
    plt.show()


def train():
    # Split-out validation dataset
    array = dataset.values
    data = array[:, 0:4]
    labels = array[:, 4]
    data_train, data_validation, labels_train, labels_validation = train_test_split(
        data, labels, test_size=0.20, random_state=1)

    # Make predictions on validation dataset
    model = SVC(gamma='auto')
    model.fit(data_train, labels_train)
    predictions = model.predict(data_validation)

    # Evaluate predictions
    print(accuracy_score(labels_validation, predictions))
    print(confusion_matrix(labels_validation, predictions))
    print(classification_report(labels_validation, predictions))


if __name__ == '__main__':
    # plot()
    train()
