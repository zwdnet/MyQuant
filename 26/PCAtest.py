# coding:utf-8
"""用降维算法中的PCA算法将鸢尾花数据降维并可视化。"""


import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplotlib as plt


if __name__ == "__main__":
    data = load_iris()
    y = data.target
    x = data.data
    # 四维数据降为两维
    pca = PCA(n_components = 2)
    reduced_x = pca.fit_transform(x)
    # 按类别保存降维后数据
    red_x, red_y = [], []
    blue_x, blue_y = [], []
    green_x, green_y = [], []
    
    for i in range(len(reduce_x)):
        if y[i] == 0:
            red_x.append(reduced_x[i][0])
            red_y.append(reduced_x[i][1])
        elif y[i] == 1:
            blue_x.append(reduced_x[i][0])
            blue_y.append(reduced_x[i][1])
        else:
            green_x.append(reduced_x[i][0])
            green_y.append(reduced_x[i][1])
            
    # 绘图
    plt.scatter(red_x, red_y, c = "r", marker = "x")
    plt.scatter(blue_x, blue_y, c = "b", marker = "D")
    plt.scatter(green_x, green_y, c = "g", marker = ".")
    plt.savefig("iris.png")
    