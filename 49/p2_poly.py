# coding:utf-8
# 机器学习A-Z
# 第二部分 多项式线性回归


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer,LabelEncoder, OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    dataset = pd.read_csv("./data/Machine Learning A-Z Chinese Template Folder/Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv")
    print(dataset)
    x = dataset.iloc[:, 1].values
    y = dataset.iloc[:, 2].values
    plt.figure()
    plt.scatter(x, y)
    plt.savefig("poly_data.png")
    plt.close()
    # 进行多项式回归
    lin_reg = LinearRegression()
    lin_reg.fit(x.reshape(-1, 1), y)
    poly_reg = PolynomialFeatures(degree = 2)
    x_poly = poly_reg.fit_transform(x.reshape(-1, 1))
    print(x_poly)
    print(x, x.reshape(-1, 1))
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(x_poly, y)
    poly_reg_3 = PolynomialFeatures(degree = 3)
    x_poly_3 = poly_reg_3.fit_transform(x.reshape(-1, 1))
    print(x_poly_3)
    lin_reg_3 = LinearRegression()
    lin_reg_3.fit(x_poly_3, y)
    poly_reg_4 = PolynomialFeatures(degree = 4)
    x_poly_4 = poly_reg_4.fit_transform(x.reshape(-1, 1))
    print(x_poly_4)
    lin_reg_4 = LinearRegression()
    lin_reg_4.fit(x_poly_4, y)
    # 绘图显示结果
    plt.figure()
    plt.scatter(x, y, color = "red")
    plt.plot(x, lin_reg.predict(x.reshape(-1, 1)), color = "blue")
    plt.plot(x, lin_reg_2.predict(x_poly), color = "green")
    plt.plot(x, lin_reg_3.predict(x_poly_3), color = "black")
    plt.plot(x, lin_reg_4.predict(x_poly_4), color = "magenta")
    plt.title("Truth or Bluff")
    plt.xlabel("Position Level")
    plt.ylabel("Salary")
    plt.savefig("PolyRegression.png")
    plt.close()
    # 可视化预测结果
    plt.figure()
    plt.scatter(x, y, color = "red")
    x_grid = np.arange(min(x), max(x), 0.1)
    x_grid = x_grid.reshape(len(x_grid), 1)
    plt.plot(x_grid, lin_reg_4.predict(poly_reg_4.fit_transform(x_grid)), color = "magenta")
    plt.title("Truth or Bluff(4th poly regression)")
    plt.xlabel("Position Level")
    plt.ylabel("Salary")
    plt.savefig("PolyRegression4.png")
    plt.close()
    