# coding:utf-8
# 机器学习A-Z
# 第二部分 简单线性回归


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer,LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    dataset = pd.read_csv("./data/Machine Learning A-Z Chinese Template Folder/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv")
    print(dataset)
    # 划分训练集和测试集
    x = dataset.YearsExperience.values.reshape(-1, 1)
    y = dataset.Salary.values.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)
    print(x_train, y_train, x_test, y_test)
    print(len(x_train), len(y_train), len(x_test), len(y_test))
    # 进行线性回归
    regress = LinearRegression()
    regress.fit(x_train, y_train)
    # 用模型进行预测，用测试集来预测
    y_pred = regress.predict(x_test)
    print(y_pred, y_test)
    # 结果绘图
    # 训练集结果
    plt.figure()
    plt.scatter(x_train, y_train, color = "red")
    plt.plot(x_train, regress.predict(x_train), color = "blue")
    plt.title("Salary vs Expernce (trainning set)")
    plt.xlabel("Year of Expernce")
    plt.ylabel("Salary")
    plt.savefig("P2_train.png")
    plt.close()
    # 测试集结果
    plt.figure()
    plt.scatter(x_test, y_test, color = "red")
    plt.plot(x_train, regress.predict(x_train), color = "blue")
    plt.title("Salary vs Expernce (trainning set)")
    plt.xlabel("Year of Expernce")
    plt.ylabel("Salary")
    plt.savefig("P2_test.png")
        