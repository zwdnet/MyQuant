# coding:utf-8
# 机器学习A-Z
# 第四部分 聚类算法 k均值算法


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.preprocessing import Imputer,LabelEncoder, OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans


if __name__ == "__main__":
    dataset = pd.read_csv("./data/knn/Mall_Customers.csv")
    print(dataset.info())
    x = dataset.iloc[:, 3:5].values
    print(x)
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, max_iter = 300, n_init = 10, init = 'k-means++',  random_state = 0)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title("kmeans_wcss")
    plt.savefig("wcss.png")
    plt.figure()
    # 用组数5进行kmeans聚类
    kmeans = KMeans(n_clusters = 5, max_iter = 300, n_init = 10, init = 'k-means++',  random_state = 0)
    y_kmeans = kmeans.fit_predict(x)
    plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = "red", label = "Cluster0")
    plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = "blue", label = "Cluster1")
    plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = "green", label = "Cluster2")
    plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = "cyan", label = "Cluster3")
    plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = "magenta", label = "Cluster4")
    plt.scatter(kmeans.cluster_centers_[:, 0],  kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
    plt.savefig("knn.png")
    