# coding:utf-8
"""
kaggle泰坦尼克号竞赛第五次提交
用朴素贝叶斯
"""


import numpy as np
import pandas as pd
import tools
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB


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
	# 将年龄分段按15，55分段
	train_data.loc[(train_data.Age <= 15), "Age"] = 0
	train_data.loc[((train_data.Age > 15) and (train_data.Age <= 55)), "Age"] = 1
	train_data.loc[(train_data.Age > 55), "Age"] = 3
	test_data.loc[(test_data.Age <= 15), "Age"] = 0
	test_data.loc[((test_data.Age > 15) and (test_data.Age <= 55)), "Age"] = 1
	test_data.loc[(test_data.Age > 55), "Age"] = 3
	
	features_train = ['Pclass', 'Sex', 'Age', 'Family', 'Embarked', 'Survived', 'Cabin']
	train_data = tools.featureFind(train_data, features_train)
	features_test = ['PassengerId', 'Pclass', 'Sex', 'Age', 'Family', 'Embarked', 'Cabin']
	test_data = tools.featureFind(test_data, features_test)
	
	print(train_data.head())
	print(train_data.Family)
	
	# 进行朴素贝叶斯模型建模
	features = ['Pclass', 'Sex', 'Age', 'Family', 'Embarked', 'Cabin']
	model = MultinomialNB(alpha = 2.0)
	model.fit(train_data[features], train_data["Survived"])
	print("模型评分:", model.score(train_data[features], train_data["Survived"]))
	result = model.predict(test_data[features])
	# 输出到文件
	output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': result})
	output.to_csv("submit05.csv", index = False)
	print("结果输出完毕!")
	print(model.predict_proba(test_data[features]))
	