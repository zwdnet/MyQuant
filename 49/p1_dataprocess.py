# coding:utf-8
# 机器学习A-Z
# 第一部分 数据预处理


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer,LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    dataset = pd.read_csv("./data/Machine Learning A-Z Chinese Template Folder/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Data_Preprocessing/Data.csv")
    print(dataset)
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 3].values
    print(x, y)
    # 处理缺失数据
    imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
    imputer = imputer.fit(x[:, 1:3])
    x[:, 1:3] = imputer.transform(x[:, 1:3])
    print(x)
    # 将分类数据编码
    labelencoder_x = LabelEncoder()
    x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
    print(x)
    # 虚拟编码
    onehotEncoder = OneHotEncoder(categorical_features = [0])
    x = onehotEncoder.fit_transform(x).toarray()
    # 处理因变量，不是必要的
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)
    print(y)
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
    print(x_train, y_train, x_test, y_test)
    # 特征缩放
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.transform(x_test)
    print(x_train, x_test)
    