# coding:utf-8
# 多元线性回归实验


import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols


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
	
	