# coding:utf-8
# 分类算法中的KNN,Nbayes,决策树算法


from sklearn.neighbors import KNeighborsClassifier as KNN


if __name__ == "__main__":
    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    
    neigh = KNN(n_neighbors = 3)
    neigh.fit(X, y)
    print(neigh.predict([[1.1]]))
    
    