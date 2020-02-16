# coding:utf-8
# 多元线性回归实验


import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
# from statsmodels.stats import diagnostic as dia
import scipy.stats as stats
import statsmodels.tsa.api as smt
from statsmodels.stats.stattools import durbin_watson


if __name__  == "__main__":
	data = pd.read_csv("mtcars.csv")
	print(data.head(), data.describe(), data.columns)
	col = data.columns
	print(col[1], col[2:].values)
	Y = data[col[1]]
	X = data[col[2:].values]
	print(X, Y)
	# print(X.info(), Y.info())
	
	# 增加常数项
	X = sm.add_constant(X)
	print(X.head())
	model = sm.OLS(Y, X).fit()
	print(model.summary())
	params = model.params
	# 计算模型残差
	resid = model.resid
	plt.scatter(data["mpg"], resid)
	plt.savefig("residual.png")
	plt.close()
	resid.plot.density()
	plt.savefig("resid_density.png")
	plt.close()
	# 检测异方差性
	# het = dia.het_breuschpagan(resid,data["mpg"].values)
	# print('p-value: ', het[-1])
	print(stats.stats.spearmanr(resid.values, data["mpg"].values))
	# 计算自相关系数
	acf = smt.stattools.acf(resid.values, nlags = 5)
	print(acf)
	# 可视化
	fig = smt.graphics.plot_acf(resid, lags=5, alpha=0.5)
	fig.savefig("acf.png")
	# 用dw检验来检测自关联性
	dw = durbin_watson(resid)
	print(dw)
	