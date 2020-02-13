# coding:utf-8
# 《量化投资:以python为工具》第五部分


import mpl_finance as mf
import pandas as pd
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, date2num
from datetime import datetime
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import numpy as np


# 计算data数据的延迟量为period的动量
def momentum(data, period):
	lagPrice = data.shift(period)
	moment = data - lagPrice
	moment = moment.dropna()
	return moment
	
	
# 计算信号点预测准确率，及相应的收益率
def strat(tradeSignal, ret):
	indexDate = tradeSignal.index
	ret = ret[indexDate]
	tradeRet = ret * tradeSignal
	tradeRet[tradeRet == (-0)] = 0
	winRate = len(tradeRet[tradeRet > 0]) / len(tradeRet[tradeRet != 0])
	meanWin = sum(tradeRet[tradeRet > 0]) / len(tradeRet[tradeRet > 0])
	meanLoss = sum(tradeRet[tradeRet < 0]) / len(tradeRet[tradeRet < 0])
	perform = {"winRate":winRate,
						"meanWin":meanWin,
						"meanLoss":meanLoss}
	return perform


if __name__ == "__main__":
	register_matplotlib_converters()
	# 读取数据
	ssec2015 = pd.read_csv("ssec2015.csv")
	ssec2015 = ssec2015.iloc[:, 1:]
	print(ssec2015.head(n = 3))
	print(ssec2015.tail())
	ssec2015.Date = [date2num(datetime.strptime(date, "%Y-%m-%d")) for date in ssec2015.Date]
	ssec2015list = list()
	for i in range(len(ssec2015)):
		ssec2015list.append(ssec2015.iloc[i, :])
	print(ssec2015list[0:10])
	
	ax = plt.subplot()
	mondays = WeekdayLocator(MONDAY)
	weekFormatter = DateFormatter("%y %b %d")
	ax.xaxis.set_major_locator(mondays)
	ax.xaxis.set_major_locator(DayLocator())
	ax.xaxis.set_major_formatter(weekFormatter)
	ax.set_title("SH2015")
	mf.candlestick_ohlc(ax, ssec2015list, width = 0.7, colorup = "r", colordown = "g")
	# plt.setp(plt.gca().get_xticklabels(), rotation = 50, horizontalalignment "center")
	plt.savefig("SH2015-candle.png")
	
	# 动量交易
	print("动量交易策略")
	Wanke = pd.read_csv("Vanke.csv")
	Wanke.index = Wanke.iloc[:, 1]
	Wanke.index = pd.to_datetime(Wanke.index, format = "%Y-%m-%d")
	Wanke = Wanke.iloc[:, 2:]
	print(Wanke.head())
	
	Close = Wanke.Close
	print(Close.describe())
	# 求滞后五期收盘价
	lag5Close = Close.shift(5)
	# 求5日动量
	momentum5 = Close - lag5Close
	print(momentum5.tail())
	# 绘图
	plt.subplot(211)
	plt.plot(Close, "b*")
	plt.title("Wanke Close Price")
	plt.subplot(212)
	plt.plot(momentum5, "r-*")
	plt.title("Momentum")
	plt.savefig("moment.png")
	plt.close()
	
	momen35 = momentum(Close, 35)
	plt.plot(momen35, "g.")
	plt.savefig("moment35.png")
	plt.close()
	
	# 计算交易信号
	signal = []
	for i in momen35:
		if i > 0:
			signal.append(1)
		else:
			signal.append(-1)
	signal = pd.Series(signal, index = momen35.index)
	print(signal.head())
	# 根据信号执行交易
	tradeSig = signal.shift(1)
	ret = Close/Close.shift(1) - 1
	Mom35Ret = (ret * tradeSig).dropna()
	print(Mom35Ret[:5])
	# 计算指标策略胜率
	Mom35Ret[Mom35Ret == -0] = 0
	win = Mom35Ret[Mom35Ret > 0]
	winrate = len(win)/len(Mom35Ret[Mom35Ret != 0])
	print("动量交易策略胜率:%lf" % winrate)
	# 画收益和损失直方图
	loss = -Mom35Ret[Mom35Ret < 0]
	
	plt.subplot(211)
	win.hist()
	plt.title("win")
	plt.subplot(212)
	loss.hist()
	plt.title("lost")
	plt.savefig("win_lost.png")
	# 计算两种收益率的平均值与分位数值
	performance = pd.DataFrame({
	"win":win.describe(),
	"loss":loss.describe()})
	print(performance)
	
	# RSI指标
	print("RSI指标")
	# 读取数据
	BOCM = pd.read_csv("BOCM.csv")
	BOCM.index = BOCM.iloc[:, 1]
	BOCM.index = pd.to_datetime(BOCM.index, format = "%Y-%m-%d")
	BOCM = BOCM.iloc[:, 2:]
	print(BOCM.head())
	# 计算RSI指标
	BOCMclp = BOCM.Close
	clprcChange = BOCMclp - BOCMclp.shift(1)
	clprcChange = clprcChange.dropna()
	print(clprcChange.head())
	
	indexprc = clprcChange.index
	upPrc = pd.Series(0, index = indexprc)
	upPrc[clprcChange > 0] = clprcChange[clprcChange > 0]
	downPrc = pd.Series(0, index = indexprc)
	downPrc[clprcChange < 0] = -clprcChange[clprcChange < 0]
	rsidata = pd.concat([BOCMclp, clprcChange, upPrc, downPrc], axis = 1)
	rsidata.columns = ["Close", "PrcChange", "upPrc", "downPrc"]
	rsidata = rsidata.dropna()
	print(rsidata.head())
	
	# 计算RSI6
	SMUP = []
	SMDOWN = []
	for i in range(6, len(upPrc) - 1):
		SMUP.append(np.mean(upPrc.values[(i-6):i], dtype = np.float32))
		SMDOWN.append(np.mean(downPrc.values[(i-6):i], dtype = np.float32))
		
	rsi6 = [100*SMUP[i] / (SMUP[i]+ SMDOWN[i]) for i in range(0, len(SMUP))]
	
	indexRsi = indexprc[7:]
	Rsi6 = pd.Series(rsi6, index = indexRsi)
	print(Rsi6.head())
	
	print(Rsi6.describe())
	
	# 绘图
	UP = pd.Series(SMUP, index = indexRsi)
	DOWN = pd.Series(SMDOWN, index = indexRsi)
	plt.subplot(411)
	plt.plot(BOCMclp, "k")
	plt.subplot(412)
	plt.plot(UP, "b")
	plt.subplot(413)
	plt.plot(DOWN, "y")
	plt.subplot(414)
	plt.plot(Rsi6, "g")
	plt.savefig("RSI.png")
	plt.close()
	
	# 计算RSI24
	SMUP = []
	SMDOWN = []
	for i in range(24, len(upPrc) - 1):
		SMUP.append(np.mean(upPrc.values[(i-24):i], dtype = np.float32))
		SMDOWN.append(np.mean(downPrc.values[(i-24):i], dtype = np.float32))
		
	rsi24 = [100*SMUP[i] / (SMUP[i]+ SMDOWN[i]) for i in range(0, len(SMUP))]
	
	indexRsi = indexprc[25:]
	Rsi24 = pd.Series(rsi24, index = indexRsi)
	print(Rsi24.head())
	
	print(Rsi24.describe())
	
	# 画出交叉
	plt.plot(Rsi6[100:200])
	plt.plot(Rsi24[100:200])
	plt.legend("best")
	plt.savefig("RSICross.png")
	plt.close()
	
	"""策略为:当RSI6>80或RSI6向下穿过RSI24为卖出信号。当RSI6<20或RSI向上穿过RSI24为买入信号。"""
	# 计算交易信号
	Sig1 = []
	for i in Rsi6:
		if i > 80.0:
			Sig1.append(-1)
		elif i < 20:
			Sig1.append(1)
		else:
			Sig1.append(0)
		
	date1 = Rsi6.index
	Signal1 = pd.Series(Sig1, index = date1)
	print(Signal1[Signal1 == 1].head())
	
	# 信号2，交叉
	Signal2 = pd.Series(0, index = Rsi24.index)
	lagrsi6 = Rsi6.shift(1)
	lagrsi24 = Rsi24.shift(1)
	for i in Rsi24.index:
		if (Rsi6[i] > Rsi24[i]) & (lagrsi6[i] < lagrsi24[i]):
			Signal2[i] = 1
		if (Rsi6[i] < Rsi24[i]) & (lagrsi6[i] > lagrsi24[i]):
			Signal2[i] = 1
	
	# 将两个信号结合起来。
	signal = Signal1 + Signal2
	signal[signal >= 1] = 1
	signal[signal <= -1] = -1
	signal = signal.dropna()
	
	# 求收益率
	tradSig = signal.shift(1)
	ret = BOCMclp/BOCMclp.shift(1) - 1
	print(ret.head())
	# 买入交易收益率
	ret = ret[tradSig.index]
	buy = tradSig[tradSig == 1]
	buyRet = ret[tradSig == 1] * buy
	# 卖出交易收益率
	sell = tradSig[tradSig == -1]
	sellRet = ret[tradSig == -1] * sell
	# 合并收益率
	tradeRet = ret * tradSig
	# 画收益时序图
	plt.plot(tradeRet)
	plt.savefig("RsiReturnRate.png")
	
	BuyOnly = strat(buy, ret)
	SellOnly = strat(sell, ret)
	Trade = strat(tradSig, ret)
	Test = pd.DataFrame({
	"BuyOnly" : BuyOnly,
	"SellOnly" : SellOnly,
	"Trade" : Trade
	})
	print(Test)
	
	# 比较累积收益率
	cumStock = np.cumprod(1+ret) - 1
	cumTrade = np.cumprod(1+tradeRet) - 1
	plt.subplot(211)
	plt.plot(cumStock)
	plt.subplot(212)
	plt.plot(cumTrade)
	plt.savefig("Rsi_cumReturn.png")
	plt.close()
			