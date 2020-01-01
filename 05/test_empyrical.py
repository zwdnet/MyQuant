# coding:utf-8
from pyalgotrade import strategy
from pyalgotrade.technical import ma, cross
from pyalgotrade.barfeed import yahoofeed, quandlfeed
from pyalgotrade.stratanalyzer import returns, sharpe, drawdown, trades
import empyrical as ep
import numpy as np


class SMACrossOver(strategy.BacktestingStrategy):
	def __init__(self, feed, instrument, smaPeriod):
		super().__init__(feed)
		self.__instrument = instrument
		self.__position = None
		# self.setUseAdjustedValues(True)
		self.__prices = feed[instrument].getPriceDataSeries()
		self.__sma = ma.SMA(self.__prices, smaPeriod)
		print("初始资金量:%.2f" % self.getBroker().getCash())
		
	def getSMA(self):
		return self.__sma
		
	def onEnterCanceled(self, position):
		self.__position = None
		
	def onExitOk(self, position):
		self.__position = None
		
	def onExitCanceled(self, position):
		self.__position.exitMarket()
		
	def onBars(self, bars):
		if self.__position is None:
			if cross.cross_above(self.__prices, self.__sma) > 0:
				shares = int(self.getBroker().getCash()*0.9 / bars[self.__instrument].getPrice())
				print(shares, self.getBroker().getCash())
				self.__position = self.enterLong(self.__instrument, shares, True)
		elif not self.__position.exitActive() and cross.cross_below(self.__prices, self.__sma) > 0:
			self.__position.exitMarket()
		

if __name__ == "__main__":
	feed = quandlfeed.Feed()
	feed.addBarsFromCSV("orcl", "WIKI-ORCL-2000-quandl.csv")
	
	feedBase = quandlfeed.Feed()
	feedBase.addBarsFromCSV("ibm", "WIKI-IBM-2000-quandl.csv")
	myStrategy = SMACrossOver(feed, "orcl", 20)
	baseStrategy = SMACrossOver(feedBase, "ibm", 20)
	retAnalyzer = returns.Returns()
	myStrategy.attachAnalyzer(retAnalyzer)
	retBase = returns.Returns()
	baseStrategy.attachAnalyzer(retBase)
	sharpeRatioAnalyzer = sharpe.SharpeRatio()
	myStrategy.attachAnalyzer(sharpeRatioAnalyzer)
	drawDownAnalyzer = drawdown.DrawDown()
	myStrategy.attachAnalyzer(drawDownAnalyzer)
	
	myStrategy.run()
	baseStrategy.run()
	
	print("最终收益:%.2f" % myStrategy.getResult())
	print("累积收益率:%.2f%%" % (retAnalyzer.getCumulativeReturns()[-1]*100))
	print("基准收益率:%.2f%%" % (retBase.getCumulativeReturns()[-1]*100))
	print("夏普比例:%.2f" % sharpeRatioAnalyzer.getSharpeRatio(0.05))
	print("最大回撤:%.2f %%" % (drawDownAnalyzer.getMaxDrawDown()*100))
	print("最大回撤期间: %s" % (drawDownAnalyzer.getLongestDrawDownDuration()))
	
	# 用empyrical计算回测指标
	print("empyrical")
	# 先转换数据
	rt = retAnalyzer.getReturns()
	returnTest = [rt[i] for i in range(len(rt))]
	rb = retBase.getReturns()
	returnBase = [rb[i] for i in range(len(rb))]
	rt_a = np.array(rt)
	rb_a = np.array(rb)
	# 计算最大回测
	maxDrawDown = ep.max_drawdown(rt_a)
	print("最大回撤:%.2f" % maxDrawDown)
	# 计算α β值
	alpha, beta = ep.alpha_beta(rt_a, rb_a)
	print("α=%.2f\nβ=%.2f" % (alpha, beta))
	# 计算夏普比率
	sharpe = ep.sharpe_ratio(rt_a, risk_free = 0.05)
	print("夏普比率:%.2f" % sharpe)
	