# coding:utf-8
# 时间序列分析实操


import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import *
import statsmodels.api as sm
import math
# from sklearn.metrics import mean_squared_error


#将八位数字的日期转换为yyyy-mm-dd
def TransfDate(d):
	year = int(d/10000)
	month = int((d - year*10000)/100)
	day = int((d - year*10000 - month*100))
	date = format("%4d-%02d-%02d" % (year, month, day))
	return date
    

# 获取股票数据
def GetHistoryData(Code, BeginTime, EndTime):
	df = ts.get_k_data(Code, index = False,  start = TransfDate(BeginTime), end = TransfDate(EndTime))
	return df
	

# 计算模型预测的误差
#def pre_error(data, model):
#	rmse = math.sqrt(mean_squared_error(data, model))
#	return rmse
	
	
if __name__ == "__main__":
	# 读取历史数据，运行一次就行了。
	#df_300 = GetHistoryData("510300", 20120531, 20200131)
#	df_300.index = df_300["date"]
#	df_300 = df_300.drop(["date"], axis = 1)
#	df_300.to_csv("300.csv")
#	df_nas = GetHistoryData("513100", 20130531, 20200131)
#	df_nas.index = df_nas["date"]
#	df_nas = df_nas.drop(["date"], axis = 1)
#	df_nas.to_csv("nas.csv")
	# 从保存的csv文件里读取数据
	df_300 = pd.read_csv("300.csv", index_col = "date", parse_dates = ["date"])
	df_nas = pd.read_csv("nas.csv", index_col = "date", parse_dates = ["date"])
	print(df_300.head())
	print(df_nas.head())
	
	# 将数据可视化
	fig = plt.figure()
	df_300.plot(subplots = True)
	plt.title("300ETF")
	plt.savefig("300ETF.png")
	df_nas.plot(subplots = True)
	plt.title("nasETF")
	plt.savefig("nasETF.png")
	
	# 重采样 画月线
	fig = plt.figure()
	df_300["close"].resample("M").mean().plot(legend = True)
	plt.savefig("300ETF_month.png")
	
	# 改变百分率图形
	fig = plt.figure()
	plt.subplot(211)
	df_300.close.div(df_300.close.shift(1)).plot(figsize = (20, 8))
	plt.title("300ETF percent")
	plt.subplot(212)
	df_nas.close.div(df_nas.close.shift(1)).plot(figsize = (20, 8))
	plt.title("nasETF percent")
	plt.savefig("percent.png")
	
	# 计算收益率
	df_300["returns"] = df_300.close.pct_change().mul(100)
	df_nas["returns"] = df_nas.close.pct_change().mul(100)
	fig = plt.figure()
	plt.subplot(211)
	df_300.returns.plot(figsize = (20,6))
	plt.subplot(212)
	df_nas.returns.plot(figsize = (20,6))
	plt.savefig("returns.png")
	
	# 相继列的绝对改变
	fig = plt.figure()
	plt.subplot(211)
	df_300.close.diff().plot(figsize = (20, 6))
	plt.subplot(212)
	df_nas.close.diff().plot(figsize = (20, 6))
	plt.savefig("absdiff.png")
	
	# 比较两个序列
	# 正态化以前比较
	fig = plt.figure()
	df_300.close.plot()
	df_nas.close.plot()
	plt.legend(["300etf", "nasetf"])
	plt.savefig("compare1.png")
	# 正态化，从同一时间点开始比较
	df_300_cut = df_300.close["2013-05-31":]
	norm_300 = df_300_cut.div(df_300_cut.iloc[0]).mul(100)
	norm_nas = df_nas.close.div(df_nas.close.iloc[0]).mul(100)
	fig = plt.figure()
	norm_300.plot()
	norm_nas.plot()
	plt.legend(["300etf", "nasetf"])
	plt.savefig("compare2.png")
	# 窗口函数，90日均线
	# Rolling 相同大小和切片
	rolling_300 = df_300.close.rolling("90D").mean()
	rolling_nas = df_nas.close.rolling("90D").mean()
	fig = plt.figure()
	df_300.close.plot()
	rolling_300.plot()
	plt.savefig("rolling300.png")
	fig = plt.figure()
	df_nas.close.plot()
	rolling_nas.plot()
	plt.savefig("rollingNAS.png")
	# Expanding 包含之前所有数据
	expanding_300 = df_300.close.expanding().mean()
	expanding_nas = df_nas.close.expanding().mean()
	fig = plt.figure()
	df_300.close.plot()
	expanding_300.plot()
	plt.savefig("expanding300.png")
	fig = plt.figure()
	df_nas.close.plot()
	expanding_nas.plot()
	plt.savefig("expandingNAS.png")
	plt.close()
	# 两个指数的自相关性和部分自相关性
	# 自相关
	fig = plt.figure()
	plot_acf(df_300["close"], lags = 25, title = "300ETF")
	plt.savefig("300acf.png")
	fig = plt.figure()
	plot_acf(df_nas["close"], lags = 25, title = "nasETF")
	plt.savefig("nasacf.png")
	# 部分自相关
	fig = plt.figure()
	plot_pacf(df_300["close"], lags = 25, title = "300pETF")
	plt.savefig("300pacf.png")
	fig = plt.figure()
	plot_pacf(df_nas["close"], lags = 25, title = "naspETF")
	plt.savefig("naspacf.png")
	
	# 数据的趋势，季节性和噪音
	# plt.close()
	# 分解
	# fig = plt.figure()
	plt.rcParams["figure.figsize"] = 11,9
	decomposed_300 = sm.tsa.seasonal_decompose(df_300["close"], freq = 360)
	fig = decomposed_300.plot()
	fig.savefig("decompose_300.png")
	decomposed_nas = sm.tsa.seasonal_decompose(df_nas["close"], freq = 360)
	fig = decomposed_nas.plot()
	fig.savefig("decompose_nas.png")
	
	# 用单位根检验方法来检验两个序列是否是随机行走的
	from statsmodels.tsa.stattools import adfuller
	adf_300 = adfuller(df_300["close"])
	print("300etf的单位根检验p值=%lf" % adf_300[1])
	adf_nas = adfuller(df_nas["close"])
	print("NASetf的单位根检验p值=%lf" % adf_nas[1])
	
	# 序列和序列差分的稳定性
	fig = plt.figure()
	plt.subplot(211)
	decomposed_300.trend.plot()
	plt.subplot(212)
	decomposed_300.trend.diff().plot()
	fig.savefig("stand300.png")
	fig = plt.figure()
	plt.subplot(211)
	decomposed_nas.trend.plot()
	plt.subplot(212)
	decomposed_nas.trend.diff().plot()
	fig.savefig("standnas.png")
	
	# 建立模型预测
	# AR模型
	from statsmodels.tsa.arima_model import ARMA
	df300_model = ARMA(df_300["close"].diff().iloc[1:].values, order = (1, 0))
	df300_res = df300_model.fit()
	fig = plt.figure()
	fig = df300_res.plot_predict(start = 1000, end = 1100)
	fig.savefig("ar_300.png")
	print(df300_res.summary())
	# print("模型误差:%f" % pre_error(df_300["close"].diff().iloc[1:].values[1000:1100], df300_res.predict(start = 1000, end = 1100)))
	dfnas_model = ARMA(df_nas["close"].diff().iloc[1:].values, order = (1, 0))
	dfnas_res = dfnas_model.fit()
	fig = plt.figure()
	fig = dfnas_res.plot_predict(start = 1000, end = 1100)
	fig.savefig("ar_nas.png")
	print(dfnas_res.summary())
	
	# MA模型
	df300_ma = ARMA(df_300["close"].diff().iloc[1:].values, order = (0, 1))
	df300_res = df300_ma.fit()
	fig = plt.figure()
	fig = df300_res.plot_predict(start = 1000, end = 1100)
	fig.savefig("ma_300.png")
	print(df300_res.summary())
	# print("模型误差:%f" % pre_error(df_300["close"].diff().iloc[1:].values[1000:1100], df300_res.predict(start = 1000, end = 1100)))
	dfnas_ma = ARMA(df_nas["close"].diff().iloc[1:].values, order = (0, 1))
	dfnas_res = dfnas_ma.fit()
	fig = plt.figure()
	fig = dfnas_res.plot_predict(start = 1000, end = 1100)
	fig.savefig("ma_nas.png")
	print(dfnas_res.summary())
	
	# ARMA模型
	df300_arma = ARMA(df_300["close"].diff().iloc[1:].values, order = (3, 3))
	df300_res = df300_arma.fit()
	fig = plt.figure()
	fig = df300_res.plot_predict(start = 1000, end = 1100)
	fig.savefig("arma_300.png")
	print(df300_res.summary())
	# print("模型误差:%f" % pre_error(df_300["close"].diff().iloc[1:].values[1000:1100], df300_res.predict(start = 1000, end = 1100)))
	#dfnas_arma = ARMA(df_nas["close"].diff().iloc[1:].values, order = (3, 3))
#	dfnas_res = dfnas_arma.fit()
#	fig = plt.figure()
#	fig = dfnas_res.plot_predict(start = 1000, end = 1100)
#	fig.savefig("arma_nas.png")
#	print(dfnas_res.summary())

	# ARIMA模型
	from statsmodels.tsa.arima_model import ARIMA
	df300_arima = ARIMA(df_300["close"].diff().iloc[1:].values, order = (2, 1, 0))
	df300_res = df300_arima.fit()
	fig = plt.figure()
	fig = df300_res.plot_predict(start = 1000, end = 1100)
	fig.savefig("arima_300.png")
	print(df300_res.summary())
	
	# VAR模型
	#train_sample = pd.concat([norm_300.diff().iloc[1:], norm_nas.diff().iloc[1:]], axis = 1)
#	model = sm.tsa.VARMAX(train_sample, order = (2, 1), trend = "c")
#	result = model.fit(maxiter = 1000, disp = True)
#	print(result.summary())
#	fig = result.plot_diagnostics()
#	fig.savefig("var_dio.png")
#	pre_res = result.predict(start = 1000, end = 1100)
#	fig = plt.figure()
#	plt.plot(pre_res)
#	fig.savefig("var_pre.png")
	
	# SARIMA模型
#	train_sample = df_300["close"].diff().iloc[1:].values
#	model = sm.tsa.SARIMAX(train_sample, order = (4, 0, 4), trend = "c")
#	result = model.fit(maxiter = 1000, disp = True)
#	print(result.summary())
#	fig = plt.figure()
#	plt.plot(train_sample[1:600], color = "red")
#	plt.plot(result.predict(start = 0, end = 600), color = "blue")
#	fig.savefig("SARIMA.png")
	
	# 未观察成分模型
	train_sample = df_300["close"].diff().iloc[1:].values
	model = sm.tsa.UnobservedComponents(train_sample, "local level")
	result = model.fit(maxiter = 1000, disp = True)
	print(result.summary())
	fig = plt.figure()
	plt.plot(train_sample[1:600], color = "red")
	plt.plot(result.predict(start = 0, end = 600), color = "blue")
	fig.savefig("unobserve.png")
	
	# 动态因子模型
	train_sample = pd.concat([norm_300.diff().iloc[1:], norm_nas.diff().iloc[1:]], axis = 1)
	model = sm.tsa.DynamicFactor(train_sample, k_factors = 1, factor_order = 2)
	result = model.fit(maxiter = 1000, disp = True)
	print(result.summary())
	predicted_result = result.predict(start = 0, end = 1000)
	fig = plt.figure()
	plt.plot(train_sample[:500], color = "red")
	plt.plot(predicted_result[:500], color = "blue")
	fig.savefig("dfmodel.png")
	