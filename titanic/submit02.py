# coding:utf-8
"""
kaggle泰坦尼克号竞赛第二次提交
参考https://blog.csdn.net/han_xiaoyang/article/details/49797143
"""


import numpy as np
import pandas as pd
import tools
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols


# 获取回归模型的几个指标
def get_index(model):
	R = model.rsquared
	adjR = model.rsquared_adj
	AIC = model.aic
	F = model.fvalue
	pF = model.f_pvalue
	res = pd.Series({"R" : R, "adjR" : adjR, "AIC" : AIC, "F" : F, "pF" : pF})
	return res


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
	LR = LinearRegression()
	# 设置进行交叉验证
	kf = KFold(5, random_state = 0)
	train_target = new_train_data["Survived"]
	accuracys = []
	for train, test in kf.split(new_train_data):
		LR.fit(new_train_data.loc[train, predictors], new_train_data.loc[train, "Survived"])
		pred = LR.predict(new_train_data.loc[test, predictors])
		pred[pred >= 0.6] = 1
		pred[pred < 0.6] = 0
		accuracy = len(pred[pred == new_train_data.loc[test, "Survived"]])/len(test)
		accuracys.append(accuracy)
	print(np.mean(accuracys))
	# 进行预测，输出提交结果。
	pred = LR.predict(test_data.loc[:, predictors])
	pred[pred >= 0.6] = 1
	pred[pred < 0.6] = 0
	output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': pred.astype(np.int16)})
	output.to_csv("submit02.csv", index = False)
	print("结果输出完毕!")
	# 输出回归结果
	print("回归系数:", LR.coef_)
	print("截距:", LR.intercept_)
	X = new_train_data[predictors]
	y = new_train_data["Survived"]
	Y = LR.predict(X)
	print("模型评分:", LR.score(X, y))
	i = 241
	for index in predictors:
		X = new_train_data[index]
		fig = plt.subplot(i)
		i += 1
		plt.plot(X, Y, "*")
		plt.plot(X, y, "o")
	plt.savefig("LRtest.png")
	
	# 看模型的假设检验
	X = new_train_data[predictors]
	X = sm.add_constant(X)
	model = sm.OLS(Y, X).fit()
	res = get_index(model)
	print("回归参数", model.params)
	print("回归结果", res)
	print(model.summary())
	