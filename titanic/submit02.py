# coding:utf-8
"""
kaggle泰坦尼克号竞赛第二次提交
参考https://blog.csdn.net/han_xiaoyang/article/details/49797143
"""


import numpy as np
import pandas as pd
import tools
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold


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
	new_train_data = tools.featureFind(train_data)
	
	print(new_train_data.head())
	
	# 线性回归模型
	# 特征变量
	predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Cabin']
	alg = LinearRegression()
	# 设置进行交叉验证
	kf = KFold(new_train_data.shape[0], n_folds = 3, random_state = 1)
	
	predictions = []
	for train, test in kf:
		# 训练数据
		train_predictors = (new_train_data[predictors].iloc[train, :])
		# 训练目标
		train_target = new_train_data["Survived"].iloc[train]
		# 应用线性回归
		alg.fit(train_predictors, train_target)
		# 用测试集进行测试
		test_predictions = alg.predict(new_train_data[predictors].iloc[test, :])
		predictions.append(test_predictions)
		
	predictions = np.concatenate(predictions, axis = 0)
	predictions[predictions > 0.5] = 1
	predictions[predictions <= 0.5] = 0
	accuracy = sum(predictions[predictions == new_train_data["Survived"]])/len(predictions)
	print(acvuracy)
	