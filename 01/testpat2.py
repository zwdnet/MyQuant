# coding:utf-8
# 测试pyalgotrade
# 参考https://zhuanlan.zhihu.com/p/28543112


from GetHistroyData import *
import pandas as pd
import tsfeed
from pyalgotrade import strategy
from pyalgotrade.technical import ma
from pyalgotrade.stratanalyzer import returns, sharpe, drawdown, trades
from pyalgotrade import plotter
from pandas.plotting import register_matplotlib_converters


class MyStrategy(strategy.BacktestingStrategy):
	def __init__(self, feed, instrument, smaPeriod):
		super(MyStrategy, self).__init__(feed, 5000)
		self.__position = None
		self.__instrument = instrument
		# self.setUseAdjustedValues(False)
		self.__sma = ma.SMA(feed[instrument].getPriceDataSeries(), smaPeriod)
		
	def onBars(self, bars):
		if self.__sma[-1] is None:
			return
			
		bar = bars[self.__instrument]
		if self.__position is None:
			if bar.getPrice() > self.__sma[-1]:
				self.__position = self.enterLong(self.__instrument, 1, True)
		elif bar.getPrice() < self.__sma[-1] and not self.__position.exitActive():
				self.__position.exitMarket()
		
	def getSMA(self):
		return self.__sma
		
	def onEnterOk(self, position):
		execInfo = position.getEntryOrder().getExecutionInfo()
		self.info("买入$%.2f" % (execInfo.getPrice()))
		
	def onEnterCanceled(self, position):
		self.__position = None
		
	def onExitOk(self, position):
		execInfo = position.getEntryOrder().getExecutionInfo()
		self.info("卖出$%.2f" % (execInfo.getPrice()))
		self.__position = None
		
	def onExitCanceled(self, position):
		self.__position.ExitMarket()


if __name__ == "__main__":
	code = "hs300"
	ktype = "M"
	start = "2005-01-01"
	end = "2017-08-01"
	frequency = 10
	# hs300 = GetHistroyData(code, start, end, ktype)
	# print(hs300.head())
	# hs300.to_csv("hs300.csv")
	df_hs300 = pd.read_csv("hs300.csv")
	start = df_hs300.date[0]
	end = df_hs300.date[len(df_hs300) - 1]
	feed = tsfeed.Feed()
	feed.addBarsFromCsv(code, "hs300.csv", start, end, ktype)
	
	myStrategy = MyStrategy(feed, "hs300", frequency)
	
	ret = returns.Returns()
	myStrategy.attachAnalyzer(ret)
	sp = sharpe.SharpeRatio()
	myStrategy.attachAnalyzer(sp)
	dd = drawdown.DrawDown()
	myStrategy.attachAnalyzer(dd)
	td = trades.Trades()
	myStrategy.attachAnalyzer(td)
	
	register_matplotlib_converters()
	plt = plotter.StrategyPlotter(myStrategy, True, False, True)
	plt.getInstrumentSubplot('hs300').addDataSeries("sma", myStrategy.getSMA())
		
	myStrategy.run()
	
	plt.savePlot("hs300.png")
	
	# 输出回测数据
	print("组合最终市值:%.2f" % myStrategy.getBroker().getEquity())
	print("组合最终市值:%.2f" % myStrategy.getResult())
	print("累计回报率%.2f%%" % (ret.getCumulativeReturns()[-1] * 100))
	print("夏普比率%.2f" % (sp.getSharpeRatio(0.05)))
	print("最大回撤率%.2f%%" % (dd.getMaxDrawDown() * 100))
	print("最大回撤时间%s" % (dd.getLongestDrawDownDuration()))
	print("交易胜率%.2f" % (float(td.getProfitableCount())/float(td.getCount())))
	