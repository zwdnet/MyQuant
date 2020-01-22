# coding:utf-8
# 《Everything you can do with a time series》程序

import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
# 上面设matplotlib的模板，对可视化时间序列数据很有用。
from pylab import rcParams
#from plotly import tools
import chart_studio.plotly as py
#from plotly.offline import init_notebook_mode, iplot
# init_notebook_mode(connected = True)
#import plotly.graph_objs as go
#import plotly.figure_factory as ff
import statsmodels.api as sm
from numpy.random import normal, seed
from scipy.stats import norm
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima_model import ARIMA
import math
# import mpl_finance as mpf
# from sklearn.metrics import mean_squared_error


if __name__ == "__main__":
	print(os.listdir("./input"))
	#——1.介绍日期与时间
	# 导入数据
	google = pd.read_csv("input/stock-time-series-20050101-to-20171231/GOOGL_2006-01-01_to_2018-01-01.csv", index_col = "Date", parse_dates = ["Date"])
	print(google.head())
	
	humidity = pd.read_csv("input/historical-hourly-weather-data/humidity.csv", index_col = "datetime", parse_dates = ['datetime'])
	print(humidity.tail())
	
	# 填充缺失值
	humidity = humidity.iloc[1:]
	humidity = humidity.fillna(method = "ffill")
	print(humidity.head())
	
	# 数据集的可视化
	fig = plt.figure()
	humidity["Kansas City"].asfreq("M").plot()
	plt.title("Humidity in Kansas City over time(Monthly frequency)")
	fig.savefig("Kansas_humidity.png")
	fig = plt.figure()
	google["2008":"2010"].plot(subplots = True, figsize = (10, 12))
	plt.title("Google stocks from 2008 to 2010")
	plt.savefig("google.png")
	
	# 时间戳
	timestamp = pd.Timestamp(2017, 1, 1, 12)
	print(timestamp)
	
	# 建立一个周期
	period = pd.Period("2017-01-01")
	print(period)
	
	# 检查一个给定的时间戳是否在一个给定的时间周期中
	print(period.start_time < timestamp < period.end_time)
	
	# 将时间戳转换为周期
	new_period = timestamp.to_period(freq = "H")
	print(new_period)
	
	# 将周期转换为时间戳
	new_timestamp = period.to_timestamp(freq = "H", how = "start")
	print(new_timestamp)
	
	# 以每天的频率建立一个时间日期索引
	dr1 = pd.date_range(start = "1/1/18", end = "1/9/18")
	print(dr1)
	
	# 以每月的频率建立一个时间日期索引
	dr2 = pd.date_range(start = "1/1/18", end = "1/1/19", freq = "M")
	print(dr2)
	
	# 不设置日期起点，设定终点和周期
	dr3 = pd.date_range(end = "1/4/2014", periods = 8)
	print(dr3)
	
	# 指定起止日期和周期
	dr4 = pd.date_range(start = "2013-04-24", end = "2014-11-27", periods = 3)
	print(dr4)
	
	# 使用to_datetime
	df = pd.DataFrame({
	"year" : [2015, 2016],
	"month" : [2, 3],
	"day" : [4, 5]
	})
	print(df)
	df = pd.to_datetime(df)
	print(df)
	df = pd.to_datetime("01-01-2017")
	print(df)
	
	# 索引变换
	fig = plt.figure()
	humidity["Vancouver"].asfreq('M').plot(legend = True)
	shifted = humidity["Vancouver"].asfreq('M').shift(10).plot(legend = True)
	shifted.legend(['Vancouver','Vancouver_lagged'])
	fig.savefig("shifted.png")
	
	# 采用气压数据演示重采样
	pressure = pd.read_csv("input/historical-hourly-weather-data/pressure.csv", index_col = "datetime", parse_dates = ["datetime"])
	print(pressure.tail())
	pressure = pressure.iloc[1:]
	# 用前值填充nan
	pressure = pressure.fillna(method = "ffill")
	print(pressure.tail())
	pressure = pressure.fillna(method = "bfill")
	print(pressure.head())
	# 输出数据规模
	print(pressure.shape)
	
	# 使用平均数从小时数据到3天数据进行向下采样
	pressure = pressure.resample("3D").mean()
	print(pressure.head())
	print(pressure.shape)
	
	# 从三日数据向每日数据进行上采样
	pressure = pressure.resample('D').pad()
	print(pressure.head())
	print(pressure.shape)
	
	#——2.金融和统计学
	# 改变的百分率
	fig = plt.figure()
	google["Change"] = google.High.div(google.High.shift())
	google["Change"].plot(figsize = (20, 8))
	fig.savefig("percent.png")
	
	# 证券收益
	fig = plt.figure()
	google["Return"] = google.Change.sub(1).mul(100)
	google["Return"].plot(figsize = (20, 8))
	fig.savefig("Return1.png")
	# 另一个计算方法
	fig = plt.figure()
	google.High.pct_change().mul(100).plot(figsize = (20, 6))
	fig.savefig("Return2.png")
	
	# 比较相继序列的绝对改变
	fig = plt.figure()
	google.High.diff().plot(figsize = (20, 6))
	fig.savefig("AbsoluteChange.png")
	
	# 比较两个不同的序列，微软和谷歌的股票
	microsoft = pd.read_csv("input/stock-time-series-20050101-to-20171231/MSFT_2006-01-01_to_2018-01-01.csv", index_col = "Date", parse_dates = ["Date"])
	# 在正态化以前绘图
	fig = plt.figure()
	google.High.plot()
	microsoft.High.plot()
	plt.legend(["Google", "Microsoft"])
	fig.savefig("Compare.png")
	
	# 进行正态化并进行比较
	normalized_google = google.High.div(google.High.iloc[0]).mul(100)
	normalized_microsoft = microsoft.High.div(microsoft.High.iloc[0]).mul(100)
	fig = plt.figure()
	normalized_google.plot()
	normalized_microsoft.plot()
	plt.legend(["Google", "Microsoft"])
	fig.savefig("NormalizedCompare.png")
	
	# Rolling窗口函数
	# 90日均线吧
	rolling_google = google.High.rolling("90D").mean()
	fig = plt.figure()
	google.High.plot()
	rolling_google.plot()
	plt.legend(["High", "Rolling Mean"])
	fig.savefig("RollongGoogle.png")
	# Expanding窗口函数
	microsoft_mean = microsoft.High.expanding().mean()
	microsoft_std = microsoft.High.expanding().std()
	fig = plt.figure()
	microsoft.High.plot()
	microsoft_mean.plot()
	microsoft_std.plot()
	plt.legend(["High", "Expanding Mean", "Expanding Standard Deviation"])
	fig.savefig("ExpandingMicrosoft.png")
	
	# 画OHLC图 调不通，pass
#	# 用mpf_finance，先准备数据
#	df = pd.DataFrame()
#	df["date"] = google.index
#	df["open"] = google.Open
#	df["high"] = google.High
#	df["low"] = google.Low
#	df["close"] = google.Close
#	# df["volume"] = google[date].Volume
#	df.index = google.index
#	print(df.describe())
#	data = [tuple(x) for x in df[['date','open','high','low','close']].values]
#	print(data[0:5])
#	fig, ax = plt.subplots()
#	mpf.candlestick_ohlc(ax, data)
#	fig.savefig("candlestick.png")

	# 自相关
	fig = plt.figure()
	plot_acf(humidity["San Diego"], lags = 25, title = "San Diego")
	plt.savefig("acf.png")
	
	# 部分自相关
	fig = plt.figure()
	plot_pacf(humidity["San Diego"], lags = 25, title = "San Diego, pacf")
	plt.savefig("pacf.png")
	plot_pacf(microsoft["Close"], lags = 25)
	plt.savefig("ms_pacf.png")
	
	# 3.时间序列分解与随机行走
	# 趋势，季节性和噪音
	fig = plt.figure()
	google["High"].plot(figsize = (16, 8))
	fig.savefig("google_trend.png")
	
	# 分解
	rcParams["figure.figsize"] = 11, 9
	decomposed_google_volume = sm.tsa.seasonal_decompose(google["High"], freq = 360)
	fig = decomposed_google_volume.plot()
	fig.savefig("decomposed.png")
	
	# 白噪音
	fig = plt.figure()
	rcParams["figure.figsize"] = 16, 6
	white_noise = np.random.normal(loc = 0, scale = 1, size = 1000)
	plt.plot(white_noise)
	fig.savefig("whitenoise.png")
	# 绘制白噪音的自相关关系
	fig = plt.figure()
	plot_acf(white_noise, lags = 20)
	plt.savefig("wn_acf.png")
	
	# 随机行走 
	# 单位根检验谷歌和微软的成交量
	adf = adfuller(microsoft["Volume"])
	print("p-value of microsoft: {}".format(float(adf[1])))
	adf = adfuller(google["Volume"])
	print("p-value of google: {}".format(float(adf[1])))
	
	# 产生一个随机行走
	fig = plt.figure()
	seed(42)
	rcParams["figure.figsize"] = 16, 6
	random_walk = normal(loc = 0, scale = 0.01, size = 1000)
	plt.plot(random_walk)
	fig.savefig("random_walk.png")
	
	fig = plt.figure()
	plt.hist(random_walk)
	fig.savefig("random_hist.png")
	
	# 稳定性
	# 初始的非稳定序列
	fig = plt.figure()
	decomposed_google_volume.trend.plot()
	fig.savefig("nonstationary.png")
	# 新的稳定的序列，即一阶差分
	fig = plt.figure()
	decomposed_google_volume.trend.diff().plot()
	fig.savefig("stationary.png")
	
	# 4.使用statstools建模
	# AR(1)模型
	# AR(1) MA(1)模型: AR参数 = 0.9
	fig = plt.figure()
	rcParams['figure.figsize'] = 16, 12
	plt.subplot(4,1,1)
	ar1 = np.array([1, -0.9])
	ma1 = np.array([1])
	AR1 = ArmaProcess(ar1, ma1)
	sim1 = AR1.generate_sample(nsample = 1000)
	plt.title("AR(1) model : AR parameter = +0.9")
	plt.plot(sim1)
	# AR(1) MA(1)模型: AR参数 = -0.9
	plt.subplot(4,1,2)
	ar2 = np.array([1, 0.9])
	ma2 = np.array([1])
	AR2 = ArmaProcess(ar2, ma2)
	sim2 = AR2.generate_sample(nsample = 1000)
	plt.title("AR(1) model : AR parameter = -0.9")
	plt.plot(sim2)
	# AR(2) MA(1)模型: AR参数 = 0.9
	plt.subplot(4,1,3)
	ar3 = np.array([2, -0.9])
	ma3 = np.array([1])
	AR3 = ArmaProcess(ar3, ma3)
	sim3 = AR3.generate_sample(nsample = 1000)
	plt.title("AR(2) model : AR parameter = +0.9")
	plt.plot(sim3)
	# AR(2) MA(1)模型: AR参数 = -0.9
	plt.subplot(4,1,4)
	ar4 = np.array([2, 0.9])
	ma4 = np.array([1])
	AR4 = ArmaProcess(ar4, ma4)
	sim4 = AR4.generate_sample(nsample = 1000)
	plt.title("AR(2) model : AR parameter = -0.9")
	plt.plot(sim4)
	fig.savefig("AR.png")
	# 预测模型
	model = ARMA(sim1, order=(1, 0))
	result = model.fit()
	print(result.summary())
	print("μ = {}, φ = {}".format(result.params[0], result.params[1]))
	# 用模型预测
	fig = plt.figure()
	fig = result.plot_predict(start = 900, end = 1010)
	fig.savefig("AR_predict.png")
	
	rmse = math.sqrt(mean_squared_error(sim1[900:1011], result.predict(start = 900, end = 999)))
	print("The root mean squared error is {}.".format(rmse))
	
	# 预测蒙特利尔的湿度
	humid = ARMA(humidity["Montreal"].diff().iloc[1:].values, order = (1, 0))
	res = humid.fit()
	fig = plt.figure()
	fig = res.plot_predict(start = 1000, end = 1100)
	fig.savefig("humid_arma.png")
	
	# 预测谷歌的收盘价
	humid = ARMA(google["Close"].diff().iloc[1:].values, order = (1, 0))
	res = humid.fit()
	fig = plt.figure()
	fig = res.plot_predict(start = 900, end = 1100)
	fig.savefig("google_arma.png")
	
	# MA(1)模拟模型
	rcParams["figure.figsize"] = 16, 6
	ar1 = np.array([1])
	ma1 = np.array([1, -0.5])
	MA1 = ArmaProcess(ar1, ma1)
	sim1 = MA1.generate_sample(nsample = 1000)
	plt.plot(sim1)
	plt.savefig("ma1.png")
	
	# 建立MA模型的预测
	model = ARMA(sim1, order=(0, 1))
	result = model.fit()
	print(result.summary())
	print("μ={} ,θ={}".format(result.params[0],result.params[1]))
	
	# 使用MA模型进行预测
	model = ARMA(humidity["Montreal"].diff().iloc[1:].values, order=(0, 3))
	result = model.fit()
	print(result.summary())
	print("μ={} ,θ={}".format(result.params[0],result.params[1]))
	result.plot_predict(start = 1000, end = 1100)
	plt.savefig("ma_forcast.png")
	
	rmse = math.sqrt(mean_squared_error(humidity["Montreal"].diff().iloc[1000:1101].values, result.predict(start=1000,end=1100)))
	print("The root mean squared error is {}.".format(rmse))
	
	# 模拟和预测微软股票的市值
	model = ARMA(microsoft["Volume"].diff().iloc[1:].values, order = (3, 3))
	result = model.fit()
	print(result.summary())
	print("μ={}, ϕ={}, θ={}".format(result.params[0],result.params[1],result.params[2]))
	result.plot_predict(start = 1000, end = 1100)
	plt.savefig("arma_forcast.png")
	
	rmse = math.sqrt(mean_squared_error(microsoft["Volume"].diff().iloc[1000:1101].values, result.predict(start=1000,end=1100)))
	print("The root mean squared error is {}.".format(rmse))
	
	# 使用ARIMA模型进行预测
	# 预测微软股票的市值
	rcParams["figure.figsize"] = 16, 6
	model = ARIMA(microsoft["Volume"].diff().iloc[1:].values, order = (2, 1, 0))
	result = model.fit()
	print(result.summary())
	result.plot_predict(start = 700, end = 1000)
	plt.savefig("Arima_predict.png")
	
	rmse = math.sqrt(mean_squared_error(microsoft["Volume"].diff().iloc[700:1001].values, result.predict(start=700,end=1000)))
	print("The root mean squared error is {}.".format(rmse))

	# VAR模型
	# 预测谷歌和微软的收盘价
	train_sample = pd.concat([google["Close"].diff().iloc[1:], microsoft["Close"].diff().iloc[1:]], axis=1)
	model = sm.tsa.VARMAX(train_sample, order = (2, 1), trend = 'c')
	result = model.fit(maxiter = 1000, disp = True)
	print(result.summary())
	predicted_result = result.predict(start = 0, end = 1000)
	fig = result.plot_diagnostics()
	fig.savefig("Var_predict.png")
	# 计算误差
	rmse = math.sqrt(mean_squared_error(train_sample.iloc[1:1002].values, predicted_result.values))
	print("The root mean squared error is {}.".format(rmse))
	
	# SARIMA模型
	# 预测谷歌的收盘价
	train_sample = google["Close"].diff().iloc[1:].values
	model = sm.tsa.SARIMAX(train_sample, order = (4, 0, 4), trend = 'c')
	result = model.fit(maxiter = 1000, disp = True)
	print(result.summary())
	predicted_result = result.predict(start = 0, end = 500)
	fig = result.plot_diagnostics()
	fig.savefig("sarimax.png")
	# 计算误差
	rmse = math.sqrt(mean_squared_error(train_sample[1:502], predicted_result))
	print("The root mean squared error is {}.".format(rmse))
	
	fig = plt.figure()
	plt.plot(train_sample[1:502], color = "red")
	plt.plot(predicted_result, color = "blue")
	plt.legend(["Actual", "Predicted"])
	plt.title("Google closing price")
	fig.savefig("sarimax_test.png")
	
	# 未观察成分模型
	# 预测谷歌的收盘价
	train_sample = google["Close"].diff().iloc[1:].values
	model = sm.tsa.UnobservedComponents(train_sample, "local level")
	result = model.fit(maxiter = 1000, disp = True)
	print(result.summary())
	predicted_result = result.predict(start = 0, end = 500)
	fig = result.plot_diagnostics()
	fig.savefig("unobserve.png")
	# 计算误差
	rmse = math.sqrt(mean_squared_error(train_sample[1:502], predicted_result))
	print("The root mean squared error is {}.".format(rmse))
	
	fig = plt.figure()
	plt.plot(train_sample[1:502], color = "red")
	plt.plot(predicted_result, color = "blue")
	plt.legend(["Actual", "Predicted"])
	plt.title("Google closing price")
	fig.savefig("unobserve_test.png")
	
	# 动态因子模型
	# 预测谷歌的收盘价
	train_sample = pd.concat([google["Close"].diff().iloc[1:], microsoft["Close"].diff().iloc[1:]], axis=1)
	model = sm.tsa.DynamicFactor(train_sample, k_factors=1, factor_order=2)
	result = model.fit(maxiter = 1000, disp = True)
	print(result.summary())
	predicted_result = result.predict(start = 0, end = 1000)
	fig = result.plot_diagnostics()
	fig.savefig("DynamicFactor.png")
	# 计算误差
	rmse = math.sqrt(mean_squared_error(train_sample[1:502], predicted_result))
	print("The root mean squared error is {}.".format(rmse))
	