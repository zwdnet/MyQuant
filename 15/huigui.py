# coding:utf-8
# 机器学习:一元线性回归


import numpy as np
import matplotlib.pyplot as plt
# from sklearn import linear_model
import pandas as pd


if __name__ == "__main__":
	x = [3.4, 1.8, 4.6, 2.3, 3.1, 5.5, 0.7, 3, 2.6, 4.3, 2.1, 1.1, 6.1, 4.8, 3.8]
	y = [26.2, 17.8, 31.3, 23.1, 27.5, 36, 14.1, 22.3, 19.6, 31.3, 24, 17.3, 43.2, 36.4, 26.1]
	x = np.array(x)
	y = np.array(y)
	print(len(x), len(y))
	plt.scatter(x, y)
	plt.savefig("scatter.png")
	plt.close()
	# 建立线性回归模型
	# df = pd.DataFrame()
	# df["X"] = x
	# df["Y"] = y
	# print(df)
	#regr = linear_model.LinearRegression()
#	# 拟合
#	regr.fit(x.reshape(-1, 1), y)
#	# 得到回归参数的二乘法估计
#	a, b = regr.coef_, regr.intercept_
#	print(a, b)
#	# 画出拟合的直线
#	yp = a*x + b
#	plt.scatter(x, y)
#	plt.plot(x, yp)
#	plt.savefig("fit.png")
#	res = regr.score(x.reshape(-1, 1), y)
#	print(res)
	
	# 用statsmodels来做
	import statsmodels.api as sm
	X = sm.add_constant(x)
	model = sm.OLS(y, X)
	result = model.fit()
	print("statsmodels做线性回归")
	res = result.params
	a = res[0]
	b = res[1]
	print(res)
	print(result.summary())
	residual = []
	for i in range(len(x)):
		cha = a + b*x[i] - y[i]
		residual.append(cha)
	print(residual)
	plt.scatter(x, residual)
	plt.savefig("residual.png")
	# 模型应用
	print(a, b)
	# 求值
	print(a + b*5.0)
	# y的95%置信区间
	
