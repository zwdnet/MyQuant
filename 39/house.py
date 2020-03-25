# coding:utf-8
# kaggle题目房价预测
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques


import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# 将训练数据和训练数据合并到一起
def concat_df(train_data, test_data):
	test_data["SalePrice"] = 0.0
	return pd.concat([train_data, test_data], sort = True).reset_index(drop = True)
    
    
# 将数据集重新分割为训练集和测试集
def divide_df(all_data):
    return all_data.loc[:1459], all_data.loc[1460:].drop(["SalePrice"], axis = 1)


if __name__ == "__main__":
	# 加载数据
	train_df = pd.read_csv("./data/train.csv")
	test_df = pd.read_csv("./data/test.csv")
	print("训练集")
	print(train_df.info())
	print("测试集")
	print(test_df.info())
	# 将训练集与测试集数据合并
	all_df = concat_df(train_df, test_df)
	all_df_backup = all_df.copy(deep = True)
	print(all_df.info())
	# 数据处理
	# 丢弃所有有缺失值的特征
	all_df = all_df.dropna(axis = 1)
	print(all_df.info())
	# 将数据拆分
	train_df, test_df = divide_df(all_df)
	print("训练集:", train_df.info())
	print("测试集:", test_df.info())
	# 建模，用多元线性回归。
	features = train_df.columns
	# features.remove("SalePrice")
	# features.remove("Id")
	X = train_df.loc[:, ["LotArea", "MiscVal"]]
	Y = train_df.loc[:, "SalePrice"]
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=532)
	
	linreg = LinearRegression()
	# 训练
	model = linreg.fit(X_train, Y_train)
	# 建模参数
	print("模型参数:", model)
	print("模型截距:", linreg.intercept_)
	print("参数权重:", linreg.coef_)
	print("模型评分:", model.score(X_test, Y_test))
	# 预测
	y_pred = linreg.predict(X_test)
	# 画图看看
	plt.figure()
	id = np.arange(len(y_pred))
	plt.plot(id, Y_test)
	plt.scatter(id, y_pred)
	plt.savefig("simplestResult.png")
	plt.close()
	# 生成提交文件
	X_test = test_df.loc[:, ["LotArea", "MiscVal"]]
	y = linreg.predict(X_test)
	Id = []
	for x in range(1461, 2920):
		Id.append(x)
	res = pd.DataFrame({"Id":Id, "SalePrice":y})
	res.to_csv("first.csv")
