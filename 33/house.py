# coding:utf-8
# 用线性回归算法预测房价


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures


if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    print(data.head())
    x = np.array(data["area"])
    y = np.array(data["price"])
    minX = min(x)
    maxX = max(x)
    length = len(x)
    print(x.shape, y.shape) 
    print(x, x.reshape(-1, length))
    plt.figure()
    plt.scatter(x, y)
    plt.savefig("scatter.png")
    # 建模
    # linear = linear_model.LinearRegression()
    # linear.fit(x.reshape(length, -1), y)
    poly_reg = PolynomialFeatures(degree = 2)
    X_poly = poly_reg.fit_transform(x.reshape([length, 1]))
    print(X_poly)
    lin_reg2 = linear_model.LinearRegression()
    lin_reg2.fit(X_poly, y)
    # 获取回归参数
    coef = linear.coef_
    inter = linear.intercept_
    print(coef, inter)
    # 可视化
    y = lin_reg2.predict(poly_reg.fit_transform(x.reshape([length, 1])))
    print(x.shape, y.shape)
    plt.plot(x, y, color = "black")
    plt.savefig("result.png")
    
