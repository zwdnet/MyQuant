# coding:utf-8
# SVM支持向量机实践


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from matplotlib.colors import ListedColormap


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
    classifier.fit(train_data, train_label.ravel())
    
    # 计算分类准确率
    print("训练集:", classifier.score(train_data, train_label))
    print("测试集:", classifier.score(test_data, test_label))
    
    # 查看决策函数
    print("训练决策函数:", classifier.decision_function(train_data))
    print("预测结果:", classifier.predict(train_data))
    
    # 绘图
    fig = plt.figure()
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
    grid_test = np.stack((x1.flat, x2.flat), axis = 1)
    
    # 设置颜色
    cm_light = ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = ListedColormap(['g','r','b'])
    grid_hat = classifier.predict(grid_test)
    grid_hat = grid_hat.reshape(x1.shape)
    # 绘图
    plt.pcolormesh(x1, x2, grid_hat, cmap = cm_light)
    plt.scatter(x[:, 0], x[:, 1], c = y[:, 0], s = 30, cmap = cm_dark)
    plt.scatter(test_data[:, 0], test_data[:, 1], c = test_label[:, 0], s = 30, edgecolors = "k", zorder = 2, cmap = cm_dark)
    plt.xlabel("length")
    plt.ylabel("width")
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.savefig("result.png")

    # 试一下将四列数据都选入模型
    # 划分数据与标签
    x, y = np.split(data, indices_or_sections = (4,), axis = 1)
    train_data, test_data, train_label, test_label = train_test_split(x, y, random_state = 1, train_size = 0.6, test_size = 0.4)
    print("训练集大小:", train_data.shape)
    print(train_data)
    print(test_data)
    
    # 训练svm分类器
    classifier = svm.SVC(C = 2, kernel = "rbf", gamma = 10, decision_function_shape = "ovr") #ovr 一对多策略
    classifier.fit(train_data, train_label.ravel())
    
    # 计算分类准确率
    print("四列数据都进行建模的结果")
    print("训练集:", classifier.score(train_data, train_label))
    print("测试集:", classifier.score(test_data, test_label))
    