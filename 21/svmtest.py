# coding:utf-8
# SVM支持向量机实践


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


# 转换类别
def Iris_label(s):
    it = {b'Iris-setosa':0, b'Iris-versicolor':1, b'Iris-virginica':2}
    return it[s]


if __name__ == "__main__":
    # 读取数据
    data = np.loadtxt("iris.data", dtype = float, delimiter = ',', converters = {4 : Iris_label})
    
    print(data)
    
    # 划分数据与标签
    x, y = np.split(data, indices_or_sections = (4,), axis = 1)
    x = x[:, 0:2] # 为了画图，只选前两列
    train_data, test_data, train_label, test_label = train_test_split(x, y, random_state = 1, train_size = 0.6, test_size = 0.4)
    print(train_data)
    print(test_data)
    