# coding:utf-8
# kaggle泰坦尼克号竞赛第一次提交
# 按https://www.kaggle.com/alexisbcook/titanic-tutorial 教程来


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import tools


if __name__ == "__main__":
	# 读取数据
	traindata, testdata = tools.loadData()
	print(train_data.head())
	print(test_data.head())
	
	# 计算男女乘客的生存率
	women = train_data.loc[train_data.Sex == 'female']["Survived"]
	rate_women = sum(women)/len(women)
	print("% of women who survied:", rate_women)
	men = train_data.loc[train_data.Sex == 'male']["Survived"]
	rate_men = sum(men)/len(men)
	print("% of men who survied:", rate_men)
	
	# 使用随机森林算法预测
	y = train_data["Survived"]
	features = ["Pclass", "Sex", "SibSp", "Parch"]
	X = pd.get_dummies(train_data[features])
	X_test = pd.get_dummies(test_data[features])
	
	model = RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state = 1)
	model.fit()
	predictions = model.predict(X_test)
	
	output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
	output.to_csv("submit01.csv", index = False)
	print("结果输出完毕!")
	