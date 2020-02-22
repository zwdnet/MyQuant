# coding:utf-8
"""
kaggle泰坦尼克号竞赛第三次提交
用逻辑回归
"""


import numpy as np
import pandas as pd
import tools
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import KFold, cross_val_score


if __name__ == "__main__":
	# 读取数据
	train_data, test_data = tools.loadData()
	print(train_data.head())
	print(test_data.head())
	
	# 输出数据
	print(train_data.info())
	print(test_data.info())
	# 探索数据
	tools.exploreData(train_data)
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
	
	# 画Sigmoid函数图像
	fig = plt.figure()
	x = np.linspace(-5, 5, 100)
	y = 1.0/(1.0 + np.exp(-x))
	plt.plot(x, y)
	plt.savefig("Sigmoid.png")
	plt.close()
	
	# 开始建模，用逻辑回归
	kf = KFold(5, random_state = 0)
	predictors = ['Pclass', 'Sex', 'Age', 'Family', 'Embarked', 'Cabin']
	lr = LR(C = 0.1, solver = "liblinear", penalty = "l2")
	lr.fit(train_data[predictors], train_data["Survived"])
	print(cross_val_score(lr, train_data[predictors], train_data["Survived"], cv = kf).mean())
	accuracys = []
	testLR = LR(C = 0.1, solver = "liblinear", penalty = "l2")
	for train, test in kf.split(train_data):
		testLR.fit(train_data.loc[train, predictors], train_data.loc[train, "Survived"])
		pred = testLR.predict_proba(train_data.loc[test, predictors])
		# print(pred.shape)
		new_pred = pred[:, 1]
		new_pred[new_pred >= 0.5] = 1
		new_pred[new_pred < 0.5] = 0
		accuracy = len(new_pred[new_pred == train_data.loc[test, "Survived"]])/len(test)
		accuracys.append(accuracy)
	print(np.mean(accuracys))
	
	# 输出预测结果提交到kaggle
	pred = lr.predict_proba(test_data.loc[:, predictors])
	new_pred = pred[:, 1]
	new_pred[new_pred >= 0.5] = 1
	new_pred[new_pred < 0.5] = 0
	output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': new_pred.astype(np.int16)})
	output.to_csv("submit03.csv", index = False)
	print("结果输出完毕!")
	
