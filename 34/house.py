# coding:utf-8
# 岭回归实例化，波士顿房价预测


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error


if __name__ == "__main__":
    # 加载数据
    data = load_boston()
    boston = pd.DataFrame(data.data, columns = data.feature_names)
    y = data.target
    boston["PRICE"] = y
    # print(boston.head())
    print(boston.info())
    # g = sns.pairplot(boston)
    # g.savefig("boston.png")
    # 分割训练测试数据
    x = boston.iloc[:, :-2].values
    y = boston["PRICE"].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
    # print(x_test.shape)
    # print(x_test[:][5].shape)
    
    # 标准化处理
    # std_x = StandardScaler()
    # x_train = std_x.fit_transform(x_train)
    # x_test = std_x.fit_transform(x_test)
    # std_y = StandardScaler()
    # y_train = std_y.fit_transform(y_train)
    # y_test = std_y.fit_transform(y_test)

    # 建模
    # 线性回归
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_lr_predict = lr.predict(x_test)
    # y_lr_predict = std_y.inverse_transform(y_lr_predict)
    # print("Lr预测值:", y_lr_predict)
    
    # 岭回归
    rd = Ridge(alpha = 0.01)
    rd.fit(x_train, y_train)
    y_rd_predict = rd.predict(x_test)
    # y_rd_predict = std_y.inverse_transform(y_rd_predict)
    # print("Ridge预测值:", y_rd_predict)
    
    # 比较两种方法
    print("lr的均方误差为：",mean_squared_error(y_test, y_lr_predict))
    print("Rd的均方误差为：",mean_squared_error(y_test, y_rd_predict))
    
    # 绘图, 横坐标为RM属性
    plt.figure()
    x_test_RM = [x[5] for x in x_test]
    print(len(x_test_RM), y_test.shape)
    plt.scatter(x_test_RM, y_test, color = "black")
    plt.plot(x_test_RM, y_lr_predict, color = "red")
    plt.plot(x_test_RM, y_rd_predict, color = "blue")
    plt.savefig("result.png")
