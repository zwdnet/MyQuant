# coding:utf-8
# 机器学习A-Z
# 第三部分 分类算法 逻辑回归


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.preprocessing import Imputer,LabelEncoder, OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix


if __name__ == "__main__":
    dataset = pd.read_csv("./data/Machine Learning A-Z Chinese Template Folder/Part 3 - Classification/Section 10 - Logistic Regression/Social_Network_Ads.csv")
    print(dataset)
    x = dataset.iloc[:, [2,3]].values
    y = dataset.iloc[:, 4].values
    # 特征缩放
    sc_x = StandardScaler()
    x = sc_x.fit_transform(x)
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
    print(x_train[:10], y_test[:10])       
    # 进行逻辑回归训练
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(x_train, y_train)
    # 用分类器预测测试集结果
    y_pred = classifier.predict(x_test)
    print(y_pred)
    # 用混淆矩阵评估分类器    
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    # 画图显示分类结果
    # 训练集
    plt.figure()
    x_set, y_set = x_train, y_train
    x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min()-1, stop = x_set[:, 0].max()+1, step = 0.01), np.arange(start = x_set[:, 1].min()-1, stop = x_set[:, 1].max()+1, step = 0.01))
    plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(('orange', 'blue'))(i), label = j)
    plt.title('Logistic Regression (Training set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.savefig("train_logistic.png")
    # 测试集
    plt.figure()
    x_set, y_set = x_test, y_test
    x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min()-1, stop = x_set[:, 0].max()+1, step = 0.01), np.arange(start = x_set[:, 1].min()-1, stop = x_set[:, 1].max()+1, step = 0.01))
    plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(('orange', 'blue'))(i), label = j)
    plt.title('Logistic Regression (Testing set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.savefig("test_logistic.png")
    