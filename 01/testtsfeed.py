# coding:utf-8
# 测试tsfeed


from pyalgotrade import strategy
import tsfeed
from pyalgotrade import plotter
from pyalgotrade.technical import ma
from pyalgotrade.technical import cross
from pandas.plotting import register_matplotlib_converters
from pyalgotrade.stratanalyzer import returns
from pyalgotrade.stratanalyzer import sharpe
from pyalgotrade.stratanalyzer import drawdown
from pyalgotrade.stratanalyzer import trades
from GetHistroyData import *


class MyStrategy(strategy.BacktestingStrategy):
	def __init__(self, feed, instrument, smaPeriod):
		super(MyStrategy, self).__init__(feed)
		self.__instrument = instrument
		self.__closed = feed[instrument].getCloseDataSeries()
		self.__ma = ma.SMA(self.__closed, smaPeriod)
		self.__position = None
		
	def getSMA(self):
		return self.__ma
		
	def onEnterLong(self, position):
		print("onEnterLong", position.getShares())
		
	def onEnterCanceled(self, position):
		self.__position = None
		print("onEnterCanceled", position.getShares())
		
	def onExitOk(self, position):
		self.__position = None
		print("onExitOk", position.getShares())
		
	def onExitCanceled(self, position):
		self.__position.exitMarket()
		print("onExitCanceled", position.getShares())
		
	def onBars(self, bars):
		if self.__position is None:
			if cross.cross_above(self.__closed, self.__ma) > 0:
				shares = int(self.getBroker().getCash() * 0.9 / bars[self.__instrument].getPrice())
				print("cross_above shares,", shares)
				self.__position = self.enterLong(self.__instrument, shares, True)
		elif not self.__position.exitActive() and cross.cross_below(self.__closed, self.__ma) > 0:
			print("cross_below")
			self.__position.exitMarket()
				
	def getClose(self):
		return self.__closed
	

if __name__ == "__main__":
	register_matplotlib_converters()
	# 按代码获取数据
	code = "603019"
	start = "2018-01-29"
	end = "2018-04-04"
	feed = tsfeed.Feed()
	feed.addBarsFromCode(code, start, end)
	myStrategy = MyStrategy(feed, code, 5)
	returnsAnalyzer = returns.Returns()
	myStrategy.attachAnalyzer(returnsAnalyzer)
	plt = plotter.StrategyPlotter(myStrategy)
	plt.getInstrumentSubplot(code).addDataSeries("SMA", myStrategy.getSMA())
	plt.getOrCreateSubplot("returns").addDataSeries("Simple returns", returnsAnalyzer.getReturns())
	myStrategy.run()
	myStrategy.info("总收益$%.2f" % myStrategy.getResult())
	plt.savePlot("tsfeed.png")
	
	# 从csv文件获取数据
	Data = GetHistroyData(code, start, end)
	Data.to_csv("data.csv")
	csvFeed = tsfeed.Feed()
	csvFeed.addBarsFromCsv(code, "data.csv", start, end)
	myStrategy2 = MyStrategy(csvFeed, code, 5)
	returnsAnalyzer2 = returns.Returns()
	myStrategy2.attachAnalyzer(returnsAnalyzer2)
	plt2 = plotter.StrategyPlotter(myStrategy2)
	plt2.getInstrumentSubplot(code).addDataSeries("SMA", myStrategy2.getSMA())
	plt2.getOrCreateSubplot("returns").addDataSeries("Simple returns", returnsAnalyzer2.getReturns())
	
	
	# 策略回测数据计算
	retAnalyzer = returns.Returns()
	myStrategy2.attachAnalyzer(retAnalyzer)
	sharpeRatioAnalyzer = sharpe.			SharpeRatio()
	myStrategy2.					attachAnalyzer(sharpeRatioAnalyzer)
	drawDownAnalyzer = drawdown.DrawDown()
	myStrategy2.attachAnalyzer(drawDownAnalyzer)
	tradesAnalyzer = trades.Trades()
	myStrategy2.attachAnalyzer(tradesAnalyzer)
	
	myStrategy2.run()
	myStrategy2.info("csv总收益$%.2f" % myStrategy2.getResult())
	plt2.savePlot("tsfeed2.png")
	
	print("Final portfolio value: $%.2f" % myStrategy.getResult())
	print("Cumulative returns: %.2f %%" % (retAnalyzer.getCumulativeReturns()[-1] * 100))
	print("Sharpe ratio: %.2f" % (sharpeRatioAnalyzer.getSharpeRatio(0.05)))
	print("Max. drawdown: %.2f %%" % (drawDownAnalyzer.getMaxDrawDown() * 100))
	print("Longest drawdown duration: %s" % (drawDownAnalyzer.getLongestDrawDownDuration()))
