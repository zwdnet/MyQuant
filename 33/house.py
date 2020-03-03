# coding:utf-8
# 用线性回归算法预测房价


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures


if __name__ == "__main__":
    # 读取数据集     
    datasets_Y = [] 
    fr = open('data.csv','r') 
    lines = fr.readlines() for line in lines:
        items = line.strip().split(',')
        datasets_X.append(int(items[0]))
        datasets_Y.append(int(items[1])
    length = len(datasets_X)
    datasets_X = np.array(datasets_X).reshape([length,1])
    datasets_Y = np.array(datasets_Y)
    minX = min(datasets_X)
    maxX = max(datasets_X)
    X = np.arange(minX,maxX).reshape([-1,1])
    poly_reg = PolynomialFeatures(degree = 2)
    X_poly = poly_reg.fit_transform(datasets_X)
    lin_reg_2 = linear_model.LinearRegression()
    lin_reg_2.fit(X_poly, datasets_Y)
    # 图像中显示 
    plt.scatter(datasets_X, datasets_Y, color = 'red')
    plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
    plt.xlabel('Area')
    plt.ylabel('Price')
    plt.savefig("result.png")
