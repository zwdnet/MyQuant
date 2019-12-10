# coding:utf-8
# 测试pyalgotrade


import tsfeed
from pyalgotrade import strategy
from pyalgotrade.stratanalyzer import returns, sharpe, drawdown, trades
from pyalgotrade import broker


# 策略类
class myStrategy(strategy.BacktestingStrategy):
	def __init__(self, feed, instrument):
		super(myStrategy, self).__init__(feed)
		self.__instrument = instrument
		self.__closed = feed[instrument].getCloseDataSeries()
		self.__position = None
		
	# 每次数据更新要执行的操作
	# 简单无脑买入100股
	def onBars(self, bars):
		bar = bars[self.__instrument]
		self.info(bar.getClose())
		self.__position = self.enterLong(self.__instrument, 100, True)
		
	# 每次买入成功后执行
	def onEnterOk(self, position):
		execInfo = position.getEntryOrder().getExecutionInfo()
		self.info("在%.2f元买入,数量%d，现金%.2f" % (execInfo.getPrice(), self.getResult(), self.getBroker().getCash()))
		
	# 买入失败后执行
	def onEnterCanceled(self, position):
		self.info("买入失败")
		self.__position = None


if __name__ == "__main__":
	# 设定数据源
	feed = tsfeed.Feed()
	feed.addBarsFromCsv("Pingan", "PA_his.csv", "2016-01-01", "2016-12-31")
	# 设定手续费
	"""
	broker_commission = broker.backtesting.TradePercentage(0.0003)
	brk = broker.backtesting.Broker(10000000, feed, broker_commission)
	"""
	ms = myStrategy(feed, "Pingan")
	
	# 添加回测指标
	retAnalyzer = returns.Returns()
	ms.attachAnalyzer(retAnalyzer)
	sharpeRatioAnalyzer = sharpe.	SharpeRatio()
	ms.attachAnalyzer(sharpeRatioAnalyzer)
	drawDownAnalyzer = drawdown.DrawDown()
	ms.attachAnalyzer(drawDownAnalyzer)
	tradesAnalyzer = trades.Trades()
	ms.attachAnalyzer(tradesAnalyzer)
	
	ms.run()
	
	# 输出回测指标
	print("策略收益%.2f" % ms.getResult())
	print("夏普比率%.2f" % (sharpeRatioAnalyzer.getSharpeRatio(0.05)))
	print("最大回撤%.2f" % (drawDownAnalyzer.getMaxDrawDown() * 100))
	print(tradesAnalyzer.getCommissionsForAllTrades())
	