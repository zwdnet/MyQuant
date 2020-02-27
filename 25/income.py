# coding:utf-8
"""用聚类算法中的k-means算法对31个省的居民消费情况进行分类。"""


import numpy as np
from sklearn.cluster import KMeans


def loadData(filePath):
    fr = open(filePath, "r+")
    lines = fr.readlines()
    retData = []
    retCityName = []
    for line in lines:
        items = line.strip().split(",")
        retCityName.append(items[0])
        retData.append([float(items[i]) for i in range(1, len(items))])
    return retData, retCityName


if __name__ == "__main__":
    data, cityName = loadData("city.txt")
    km = KMeans(n_clusters = 4)
    label = km.fit_predict(data)
    expenses = np.sum(km.cluster_centers_, axis = 1)
    print(expenses)
    CityCluster = [[], [], [], []]
    for i in range(len(cityName)):
        CityCluster[label[i]].append(cityName[i])
    for i in range(len(CityCluster)):
        print("Expenses:%.2f" % expenses[i])
        print(CityCluster[i])
