# coding:utf-8
# 用线性回归算法预测房价


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    print(data.head())
    x = data["area"].values
    y = data["price"].values
    minX = min(x)
    maxX = max(x)
    plt.figure()
    plt.scatter(x, y)
    plt.savefig("scatter.png")
    # 建模
    linear = linear_model.LinearRegression()
    linear.fit(x, y)
    # 获取回归参数
    coef = linear.coef_
    inter = linear.intercept_
    print(coef, inter)
    # 可视化
    plt.plot(x, linear.predict(x), color = "black")
    plt.savefig("result.png")
    