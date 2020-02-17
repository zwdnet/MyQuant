# coding:utf-8
# 回归模型变量选择


import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt


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
	data = pd.read_csv("vars.csv")
	# print(data.columns)
	# print(data["x2"])
	X1 = data["x1"]
	X2 = data["x2"]
	X3 = data["x3"]
	X12 = data[["x1", "x2"]]
	X13 = data[["x1", "x3"]]
	X23 = data[["x2", "x3"]]
	X123 = data[["x1", "x2", "x3"]]
	Y = data["y"]
	
	X1 = sm.add_constant(X1)
	X2 = sm.add_constant(X2)
	X3 = sm.add_constant(X3)
	X12 = sm.add_constant(X12)
	X13 = sm.add_constant(X13)
	X23 = sm.add_constant(X23)
	X123 = sm.add_constant(X123)
	
	model1 = sm.OLS(Y, X1).fit()
	# 获取几个指标
	res1 = get_index(model1)
	
	model2 = sm.OLS(Y, X2).fit()
	# 获取几个指标
	res2 = get_index(model2)
	
	model3 = sm.OLS(Y, X3).fit()
	# 获取几个指标
	res3 = get_index(model3)
	
	model4 = sm.OLS(Y, X12).fit()
	# 获取几个指标
	res4 = get_index(model4)
	
	model5 = sm.OLS(Y, X13).fit()
	# 获取几个指标
	res5 = get_index(model5)
	
	model6 = sm.OLS(Y, X23).fit()
	# 获取几个指标
	res6 = get_index(model6)
	
	model7 = sm.OLS(Y, X123).fit()
	# 获取几个指标
	res7 = get_index(model7)
	
	result = pd.DataFrame([res1, res2, res3, res4, res5, res6, res7])
	print(result)
	print(model7.params)
	print(model5.params)
	print(result[["F", "pF"]])
	print(model7.summary())
	print(model5.summary())
	print(model1.summary())
	
	yi = model7.fittedvalues
	y = Y.values
	plt.plot(y)
	plt.plot(yi, "o")
	plt.savefig("predict.png")
	