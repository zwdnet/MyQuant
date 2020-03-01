# coding:utf-8
# 用分类算法预测股市涨跌


import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import tushare as ts


if __name__ == "__main__":
    # 读取股票数据
    data = pd.read_csv("HS300_his.csv")
    print(data.head())
    data.sort_index(0,ascending=True,inplace=True) 
    print(data.head())
    dayfeature = 150
    featurenum = 4*dayfeature
    x = np.zeros((data.shape[0] - dayfeature, featurenum + 1))
    y = np.zeros((data.shape[0] - dayfeature))
    for i in range(0, data.shape[0] - dayfeature):
        x[i, 0:featurenum] = np.array(data[i:i+dayfeature][["close", "open", "low", "high"]]).reshape((1, featurenum))
        x[i, featurenum] = data.ix[i + dayfeature]["open"]
    for i in range(0, data.shape[0] - dayfeature):
        if data.ix[i + dayfeature]["close"] >= data.ix[i + dayfeature]["open"]:
            y[i] = 1
        else:
            y[i] = 0
    # 建模
    clf = svm.SVC(kernel = "rbf")
    result = [] 
    for i in range(5):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
        clf.fit(x_train, y_train)
        result.append(np.mean(y_test == clf.predict(x_test)))
    print("svm预测准确率:")
    print(result)
    
