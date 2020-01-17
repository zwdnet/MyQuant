# coding:utf-8
# 《量化投资:以python为工具》第四部分


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa import stattools
from statsmodels.tsa import arima_model
from statsmodels.graphics.tsaplots import *
from arch.unitroot import ADF
import math


if __name__ == "__main__":
	Index = pd.read_table("TRD_Index.txt", index_col = "Trddt")
	SHindex = Index[Index.Indexcd == 1]
	print(SHindex.head(3))
	print(type(SHindex))
	
	Clsindex = SHindex.Clsindex
	print(Clsindex.head(3))
	print(type(Clsindex))
	
	# 将收盘指数转换成时间序列格式
	Clsindex.index = pd.to_datetime(Clsindex.index)
	print(Clsindex.head())
	print(type(Clsindex.index))
	
	fig = plt.figure()
	Clsindex.plot()
	fig.savefig("SHindex.png")
	
	# 筛选出一定时段的数据
	SHindex.index = pd.to_datetime(SHindex.index)
	SHindexPart = SHindex["2014-10-08" : "2014-10-31"]
	print(SHindexPart.head())
	print(SHindexPart.tail())
	
	# 筛选某一特定年份的数据
	SHindex2015 = SHindex["2015"]
	print(SHindex2015.head())
	print(SHindex2015.tail(2))
	
	# 选取2015年之后的数据
	SHindexAfter2015 = SHindex["2015" : ]
	print(SHindexAfter2015.head())
	# 选取2015年之前的数据
	SHindexBefore2015 = SHindex[:"2014-12-31"]
	print(SHindexBefore2015.head())
	# 筛选某几个月的数据
	SHindex9End = SHindex["2014-09" : "2014"]
	print(SHindex9End.head())
	
	# 时间序列数据的描述性分析
	print(Clsindex.head())
	fig = plt.figure()
	Clsindex.hist()
	fig.savefig("hist.png")
	
	print("最大值:%.2f" % Clsindex.max())
	print("最小值:%.2f" % Clsindex.min())
	print("均值:%.2f" % Clsindex.mean())
	print("中位数:%.2f" % Clsindex.median())
	print("标准差:%.2f" % Clsindex.std())
	print("总结\n", Clsindex.describe())
	
	# 计算上证指数的自相关性
	data = pd.read_table("TRD_Index.txt", index_col = "Trddt")
	SHindex = data[data.Indexcd == 1]
	# 转换成时间序列类型
	SHindex.index = pd.to_datetime(SHindex.index)
	
	SHRet = SHindex.Retindex
	print(SHRet.head())
	print(SHRet.tail())
	# 计算自相关系数
	acfs = stattools.acf(SHRet, fft = False)
	print(acfs[:5])
	# 计算偏自相关系数
	pacfs = stattools.pacf(SHRet)
	print(pacfs[:5])
	fig = plt.figure()
	fig = plot_acf(SHRet, use_vlines = True, lags = 30)
	fig.savefig("acfs.png")
	fig = plot_pacf(SHRet, use_vlines = True, lags = 30)
	fig.savefig("pacfs.png")
	
	# 上证综指的平稳性
	SHClose = SHindex.Clsindex
	fig = plt.figure()
	SHClose.plot()
	fig.savefig("stabibly.png")
	fig = plt.figure()
	SHRet.plot()
	fig.savefig("SHret.png")
	fig = plt.figure()
	fig = plot_acf(SHClose, use_vlines = True, lags = 30)
	fig.savefig("SHRet_acf.png")
	
	# 单位根检验
	adfSHRet = ADF(SHRet)
	print(adfSHRet.summary().as_text())
	adfSHClose = ADF(SHClose)
	print(adfSHClose.summary().as_text())
	
	# 白噪声
	whiteNoise = np.random.standard_normal(size = 500)
	fig = plt.figure()
	plt.plot(whiteNoise, c = "b")
	fig.savefig("whiteNoise.png")
	# 上证综指的白噪声检测
	LB1 = stattools.q_stat(stattools.acf(SHRet)[1:13], len(SHRet))
	print(LB1)
	print(LB1[1][-1])
	LB2 = stattools.q_stat(stattools.acf(SHClose)[1:13], len(SHClose))
	print(LB2[1][-1])
	
	# ARMA建模
	cpi = pd.read_csv("CPI.csv", index_col = "time")
	cpi.index = pd.to_datetime(cpi.index)
	print(cpi.head())
	print(cpi.shape)
	# 训练集
	CPITrain = cpi[3:]
	# 绘制时序图
	fig = plt.figure()
	plt.plot(cpi)
	fig.savefig("timecpi.png")
	# 用单位根检验序列的稳定性
	CPITrain = CPITrain.dropna().CPI
	print(ADF(CPITrain, max_lags = 10).summary().as_text())
	# 用LB检验cpi序列是否为白噪声
	LB = stattools.q_stat(stattools.acf(CPITrain)[1:12], len(CPITrain))
	print(LB[1][-1])
	# 模型识别与估计
	fig = plt.figure()
	axe1 = plt.subplot(121)
	axe2 = plt.subplot(122)
	plot1 = plot_acf(CPITrain, lags = 30, ax = axe1)
	plot2 = plot_pacf(CPITrain, lags = 30, ax = axe2)
	fig.savefig("model.png")
	# 参数估计
	model1 = arima_model.ARIMA(CPITrain, order = (1, 0, 1)).fit()
	print(model1.summary())
	model2 = arima_model.ARIMA(CPITrain, order = (1, 0, 2)).fit()
	print(model2.summary())
	model3 = arima_model.ARIMA(CPITrain, order = (2, 0, 1)).fit()
	model4 = arima_model.ARIMA(CPITrain, order = (2, 0, 2)).fit()
	model5 = arima_model.ARIMA(CPITrain, order = (3, 0, 1)).fit()
	model6 = arima_model.ARIMA(CPITrain, order = (3, 0, 2)).fit()
	# 模型诊断
	# 计算置信区间
	print(model1.conf_int())
	print(model6.conf_int())
	# 检验残差序列是否为白噪音
	stdresid = model6.resid/math.sqrt(model6.sigma2)
	fig = plt.figure()
	plt.plot(stdresid)
	axe1 = plt.subplot(211)
	axe2 = plt.subplot(212)
	axe1 = plt.plot(stdresid)
	axe2 = plot_acf(stdresid, lags = 20)
	fig.savefig("stdresid.png")
	LB = stattools.q_stat(stattools.acf(stdresid)[1:13], len(stdresid))
	print(LB[1][-1])
	# 应用模型进行预测
	print(model6.forecast(3)[0])
	print(cpi.head(3))
	