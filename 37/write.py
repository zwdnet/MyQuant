# coding:utf-8
# KNN算法实现手写识别


from sklearn import neighbors
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # 加载数据
    digits = load_digits()
    x_data = digits.data
    y_data = digits.target
    print(x_data.shape)
    print(y_data.shape)
    
    # 划分训练测试集
    x_train, x_test, y_train, y_test =  train_test_split(x_data, y_data)
    # 训练
    knn = neighbors.KNeighborsClassifier(algorithm = "kd_tree", n_neighbors = 3)
    knn.fit(x_train, y_train)
    # 准确率评估
    predictions = knn.predict(x_test)
    print(classification_report(y_test, predictions))
    