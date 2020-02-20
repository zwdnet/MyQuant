# coding:utf-8
# 工具函数


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 读取数据
def loadData():
	train_data = pd.read_csv("./titanic/train.csv")
	test_data = pd.read_csv("./titanic/test.csv")
	return [train_data, test_data]
	
	
# 探索数据
def exploreData(data):
	# 数据可视化
	# 获救人数
	plt.subplot2grid((2, 3), (0, 0))
	data.Survived.value_counts().plot(kind = "bar")
	plt.title("Survived")
	plt.ylabel("number")
	# 乘客等级
	plt.subplot2grid((2, 3), (0, 1))
	data.Pclass.value_counts().plot(kind = "bar")
	plt.title("Class")
	plt.ylabel("number")
	# 获救者年龄分布
	plt.subplot2grid((2, 3), (0, 2))
	plt.scatter(data.Survived, data.Age)
	plt.title("Survived Age")
	plt.ylabel("number")	
	plt.grid(b = True, which = "major", axis = "y")
	# 各等级客舱年龄分布
	plt.subplot2grid((2, 3), (1, 0), colspan = 2)
	data.Age[data.Pclass == 1].plot(kind='kde')
	data.Age[data.Pclass == 2].plot(kind='kde')
	data.Age[data.Pclass == 3].plot(kind='kde')
	plt.title("Class Age")
	plt.ylabel("dense")	
	plt.legend(("1", "2", "3"), loc = "best")
	# 登船口岸人数
	plt.subplot2grid((2, 3), (1, 2))
	data.Embarked.value_counts().plot(kind = "bar")
	plt.title("Embarked")
	plt.ylabel("number")	
	
	plt.savefig("visual.png")
	plt.close()
	
	# 乘客等级与获救的关系
	fig = plt.figure()
	fig.set(alpha = 0.2)
	Survived_0 = data.Pclass[data.Survived == 0].value_counts()
	Survived_1 = data.Pclass[data.Survived == 1].value_counts()
	df = pd.DataFrame({"survive":Survived_1,"death":Survived_0})
	df.plot(kind = "bar", stacked = True)
	plt.title("class-survived")
	plt.xlabel("class")
	plt.ylabel("number")
	plt.savefig("class_survive.png")
	plt.close()
	
	# 性别与获救的关系
	fig = plt.figure()
	fig.set(alpha = 0.2)
	Survived_m = data.Survived[data.Sex == "male"].value_counts()
	Survived_f = data.Survived[data.Sex == "female"].value_counts()
	df = pd.DataFrame({"male":Survived_m,"female":Survived_f})
	df.plot(kind = "bar", stacked = True)
	plt.title("sex-survived")
	plt.xlabel("sex")
	plt.ylabel("number")
	plt.savefig("sex_survive.png")
	plt.close()
	
	# 各舱别的获救人数
	fig = plt.figure()
	fig.set(alpha = 0.65)
	plt.title("class-survive")
	
	ax1 = fig.add_subplot(141)
	data.Survived[data.Sex == "female"][data.Pclass != 3].value_counts().plot(kind = "bar", label = "female highclass", color = "green")
	ax1.set_xticklabels(["survived", "death"])
	ax1.legend(["female/high"], loc = "best")
	
	ax2 = fig.add_subplot(142, sharey = ax1)
	data.Survived[data.Sex == "female"][data.Pclass == 3].value_counts().plot(kind = "bar", label = "female low class", color = "pink")
	ax2.set_xticklabels(["death", "survived"])
	ax2.legend(["female/low"], loc = "best")
	
	ax3 = fig.add_subplot(143, sharey = ax1)
	data.Survived[data.Sex == "male"][data.Pclass != 3].value_counts().plot(kind = "bar", label = "male highclass", color = "lightblue")
	ax3.set_xticklabels(["death", "survived"])
	ax3.legend(["male/high"], loc = "best")
	
	ax4 = fig.add_subplot(144, sharey = ax1)
	data.Survived[data.Sex == "male"][data.Pclass == 3].value_counts().plot(kind = "bar", label = "male low class", color = "steelblue")
	ax4.set_xticklabels(["death", "survived"])
	ax4.legend(["male/low"], loc = "best")
	
	plt.savefig("class_sex_survive.png")
	plt.close()
	
	# 各港口登船人员的生还情况
	fig = plt.figure()
	fig.set(alpha = 0.2)
	
	Survived_0 = data.Embarked[data.Survived == 0].value_counts()
	Survived_1 = data.Embarked[data.Survived == 1].value_counts()
	df = pd.DataFrame({"survived":Survived_1, "death":Survived_0})
	df.plot(kind = "bar", stacked = True)
	plt.title("embark-survive")
	plt.xlabel("port")
	plt.ylabel("number")
	plt.savefig("embark-survive.png")
	plt.close()
	
	# 船上有亲属对死亡率的影响
	g = data.groupby(["SibSp", "Survived"])
	df = pd.DataFrame(g.count()["PassengerId"])
	print(df)
	
	g = data.groupby(["Parch", "Survived"])
	df = pd.DataFrame(g.count()["PassengerId"])
	print(df)
	
	# cabin数据是否缺失的差异
	fig = plt.figure()
	fig.set(alpha = 0.2)
	Survived_cabin = data.Survived[pd.notnull(data.Cabin)].value_counts()
	Survived_nocabin = data.Survived[pd.isnull(data.Cabin)].value_counts()
	df = pd.DataFrame({"notnull":Survived_cabin, "nulll":Survived_nocabin})
	df.plot(kind = "bar", stacked = True)
	plt.title("cabin-survive")
	plt.xlabel("cabin_null?")
	plt.ylabel("number")
	plt.savefig("cabin_survive.png")
	
	
# 清洗数据
def cleanData(train_data, test_data):
	# 清洗数据
	# 查看缺失数据
	# print(train_data.isnull().sum())
	# print(test_data.isnull().sum())
	
	# 填充缺失值，年龄用中位数
	# 登船地点用众数，Cabin因子化
	train_data["Age"].fillna(train_data["Age"].median(), inplace = True)
	test_data["Age"].fillna(test_data["Age"].median(), inplace = True)
	train_data["Embarked"] = train_data["Embarked"].fillna('S')
	train_data.loc[(train_data.Cabin.notnull()), "Cabin"] = 1
	train_data.loc[(train_data.Cabin.isnull()), "Cabin"] = 0
	test_data.loc[(test_data.Cabin.notnull()), "Cabin"] = 1
	test_data.loc[(test_data.Cabin.isnull()), "Cabin"] = 0
	print(train_data.Cabin)
	
	# 再看有无缺失数据的
	# print(train_data.isnull().sum())
	# print(test_data.isnull().sum())
	
	# 将性别数据转换为数值数据
	train_data.loc[train_data["Sex"] == "male", "Sex"] = 0
	train_data.loc[train_data["Sex"] == "female", "Sex"] = 1
	test_data.loc[test_data["Sex"] == "male", "Sex"] = 0
	test_data.loc[test_data["Sex"] == "female", "Sex"] = 1
	# 将登船地点数据转换为数值数据
	# C:0, Q:1, S:2
	train_data.loc[train_data["Embarked"] == 'C', "Embarked"] = 0
	train_data.loc[train_data["Embarked"] == 'Q', "Embarked"] = 1
	train_data.loc[train_data["Embarked"] == 'S', "Embarked"] = 2
	test_data.loc[test_data["Embarked"] == 'C', "Embarked"] = 0
	test_data.loc[test_data["Embarked"] == 'Q', "Embarked"] = 1
	test_data.loc[test_data["Embarked"] == 'S', "Embarked"] = 2
	
	# print(train_data.head())
	# print(test_data.head())
	return [train_data, test_data]
	

# 提取建模的特征
def featureFind(train_data):
	# 提取特征，构建新的训练数据
	columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Survived', 'Cabin']
	new_train_data = train_data[columns]
	# print(new_train_data.info())
	return new_train_data
