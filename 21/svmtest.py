# coding:utf-8
# SVM支持向量机实践


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm


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
    print("训练集大小:", train_data.shape)
    print(train_data)
    print(test_data)
    
    # 训练svm分类器
    classifier = svm.SVC(C = 2, kernel = "rbf", gamma = 10, decision_function_shape = "ovr") #ovr 一对多策略
    
    # 计算分类准确率
    print("训练集:", classifier.score(train_data, train_label))
    print("测试集:", classifier.score(test_data, test_label))
    
    # 查看决策函数
    print("训练决策函数:", classifier.decision_function(train_data))
    print("预测结果:", classifier.predict(train_data))
