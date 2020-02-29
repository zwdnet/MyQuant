# coding:utf-8
# 分类算法中的KNN,Nbayes,决策树算法


from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.naive_bayes import GaussianNB


if __name__ == "__main__":
    # KNN算法
    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    
    neigh = KNN(n_neighbors = 3)
    neigh.fit(X, y)
    print(neigh.predict([[1.1]]))
    
    # 决策树算法
    # 创建一颗基于基尼系数的决策树
    clf = DecisionTreeClassifier()
    data = load_iris()
    # 交叉验证
    score = cross_val_score(clf, data.data, data.target, cv = 10)
    print(score)
    
    # 贝叶斯分类
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    Y = np.array([1, 1, 1, 2, 2, 2])
    bayes = GaussianNB(priors = None)
    bayes.fit(X, Y)
    print(bayes.predict([[-0.8, -1.0]]))
    