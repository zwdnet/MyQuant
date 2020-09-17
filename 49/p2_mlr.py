# coding:utf-8
# 机器学习A-Z
# 第二部分 简单线性回归 多元线性回归


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer,LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm


if __name__ == "__main__":
    dataset = pd.read_csv("./data/Machine Learning A-Z Chinese Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv")
    print(dataset)
    print(dataset.info())
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 4].values
    print(x.shape, y.shape)
    # 将分类数据编码
    labelencoder_x = LabelEncoder()
    x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
    print(x, x.shape)
    # 虚拟编码
    onehotEncoder = OneHotEncoder(categorical_features = [3])
    x = onehotEncoder.fit_transform(x).toarray() 
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
    # 进行线性回归
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    print(y_test, y_pred)
    # 准备反向淘汰
    x_train = np.append(arr = np.ones(40, 1), values = x_train, axis = 1)
    # 进行反向淘汰
    x_opt = x_train[:, [0, 1, 2, 3, 4, 5]]
    regress_OLS = sm.OLS(endog = y_train, exdog = x_opt).fit()
    print(regress_OLS.summary())
    
    