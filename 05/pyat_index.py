# coding:utf-8
# 测试pyalgotrade回测指标是否正确
from pyalgotrade import strategy, broker
from pyalgotrade_tushare import barfeed
from pyalgotrade.bar import Frequency
from pyalgotrade.stratanalyzer import returns, sharpe, drawdown, trades
import empyrical as ep
import numpy as np


class MyStrategy(strategy.BacktestingStrategy):
	def __init__(self, feed, instrument, brk):
		super().__init__(feed, brk)
		self.__position = None
		self.__instrument = instrument
		self.__t = 0
		self.__brk = self.getBroker()
		
	def onEnterOk(self, position):
		execInfo = position.getEntryOrder().getExecutionInfo()
		self.info("BUY at $%.2f" % (execInfo.getPrice()))
		
	def onBars(self, bars):
		brk = self.getBroker()
		price = bars[self.__instrument].getPrice()
		print("times = %d price = %.2f close = %.2f" % (self.__t, price, bars[self.__instrument].getClose()))
		if self.__t == 0:
			self.__position = self.enterLong(self.__instrument, 1, True)
			print(self.__t)
		self.__t += 1
		print(self.__instrument)
		pos = brk.getPositions()
		# execInfo = pos[self.__instrument].getEntryOrder().getExecutionInfo()
		# self.info("BUY at $%.2f" % (execInfo.getPrice()))
		print(brk.getCommission().calculate(brk, price, 1))
		print("代码:%s 现金:%.2f 股票:%d" % (self.__instrument, brk.getCash(), brk.getShares(self.__instrument)))
		

# 将回测结果数据转换为数组		
def toArray(data):
	ret = []
	for i in range(len(data)):
		ret.append(data[i])
	return ret
	

# 输出数据
def output(data):
	for i in range(len(data)):
		print(data[i])
		

if __name__ == "__main__":
	feedTest = barfeed.Feed(Frequency.DAY)
	feedBase = barfeed.Feed(Frequency.DAY)
	feedTest.addBarsFromCSV("test", "test.csv")
	feedBase.addBarsFromCSV("base", "base.csv")
	
	cash = 2.5
	feerate = 0.0
	fill_stra = broker.fillstrategy.DefaultStrategy(volumeLimit=0.1)
	sli_stra = broker.slippage.NoSlippage()
	fill_stra.setSlippageModel(sli_stra)
	broker_commision = broker.backtesting.TradePercentage(feerate)
	brkTest = broker.backtesting.Broker(cash, feedTest, broker_commision)
	brkBase = broker.backtesting.Broker(cash, feedBase, broker_commision)
	brkTest.setFillStrategy(fill_stra)
	brkBase.setFillStrategy(fill_stra)
	strategyTest = MyStrategy(feedTest, "test", brkTest)
	strategyBase = MyStrategy(feedBase, "base", brkBase)
	
	retTest = returns.Returns()
	retBase = returns.Returns()
	sharpeTest = sharpe.SharpeRatio()
	sharpeBase = sharpe.SharpeRatio()
	
	strategyTest.attachAnalyzer(retTest)
	strategyTest.attachAnalyzer(sharpeTest)
	strategyBase.attachAnalyzer(retBase)
	strategyBase.attachAnalyzer(sharpeBase)
	
	strategyTest.run()
	strategyBase.run()
	
	# 输出数据
	return_test = retTest.getReturns()
	return_test_arr = toArray(return_test)
	return_base = retBase.getReturns()
	return_base_arr = toArray(return_base)
	result_test = strategyTest.getResult()
	result_base = strategyBase.getResult()
	cum_return_test = retTest.getCumulativeReturns()
	cum_return_test_arr = toArray(cum_return_test)
	cum_return_base = retBase.getCumulativeReturns()
	cum_return_base_arr = toArray(cum_return_base)
	srTest = sharpeTest.getSharpeRatio(0.036)
	srBase = sharpeBase.getSharpeRatio(0.036)
	alpha,beta = ep.alpha_beta(np.array(return_test_arr), np.array(return_base_arr))
	
	print("test分期收益率")
	output(return_test_arr)
	print("base分期收益率")
	output(return_base_arr)
	print("test策略期末收益")
	print(result_test)
	print("base策略期末收益")
	print(result_base)
	print("test累积收益率")
	output(cum_return_test_arr)
	print("base累积收益率")
	output(cum_return_base_arr)
	print("test夏普比率")
	print(srTest)
	print("base夏普比率")
	print(srBase)
	print("策略α值:%.2f β值:%.2f" % (alpha, beta))
	