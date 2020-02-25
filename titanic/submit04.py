# coding:utf-8
"""
kaggle泰坦尼克号竞赛第四次提交
用支持向量机SVM
"""


import numpy as np
import pandas as pd
import tools
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    # 读取数据
	train_data, test_data = tools.loadData()
	print(train_data.head())
	print(test_data.head())
	
	# 输出数据
	print(train_data.info())
	print(test_data.info())
	# 探索数据
	# tools.exploreData(train_data)
	# 数据清洗，特征提取
	train_data, test_data = tools.cleanData(train_data, test_data)
	# 增加一个Family字段，根据SibSp和
	# Parch之和分类
	train_data["Family"] = train_data["SibSp"] + train_data["Parch"]
	train_data.loc[(train_data.Family == 0), "Family"] = 0
	train_data.loc[((train_data.Family > 0) & (train_data.Family < 4)), "Family"] = 1
	train_data.loc[(train_data.Family >= 4), "Family"] = 2
	test_data["Family"] = test_data["SibSp"] + test_data["Parch"]
	test_data.loc[(test_data.Family == 0), "Family"] = 0
	test_data.loc[((test_data.Family > 0) & (test_data.Family < 4)), "Family"] = 1
	test_data.loc[(test_data.Family >= 4), "Family"] = 2
	features_train = ['Pclass', 'Sex', 'Age', 'Family', 'Embarked', 'Survived', 'Cabin']
	train_data = tools.featureFind(train_data, features_train)
	features_test = ['PassengerId', 'Pclass', 'Sex', 'Age', 'Family', 'Embarked', 'Cabin']
	test_data = tools.featureFind(test_data, features_test)
	
	print(train_data.head())
	print(train_data.Family)
	
	# 建模，使用SVM模型
	# 划分训练集和测试集
	predictors = ['Pclass', 'Sex', 'Age', 'Family', 'Embarked', 'Cabin']
	x = train_data[predictors]
	y = train_data["Survived"]
	train_x, train_y, x_label, y_label = train_test_split(x, y, random_state = 1, train_size = 0.6, test_size = 0.4)
	print("训练集大小:", train_x.shape)
	print("测试集大小:", train_y.shape)
	
	# 训练SVM分类器
	classifier = svm.SVC(C = 2, kernel = "linear", gamma = 10, decision_function_shape = "ovr") 
	classifier.fit(train_x, x_label)
	
	# 计算分类准确率
	print("建模的结果")
	print("训练集:", classifier.score(train_x, x_label))
	print("测试集:", classifier.score(train_y, y_label))
	
	# 预测，输出结果
	pred = classifier.predict(test_data[predictors])
	print(pred)
