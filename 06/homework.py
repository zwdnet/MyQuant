# coding:utf-8
# pyalgotrade文档，家庭作业
"""
选择四只股票，初始资金一百万，计算
其2011年的最终收益，年化收益，
 平均每日收益，每日收益标准差，夏普值
 """
 
from pyalgotrade import strategy
from pyalgotrade.barfeed import quandlfeed
from pyalgotrade.stratanalyzer import returns
from pyalgotrade.stratanalyzer import sharpe
from pyalgotrade.utils import stats
 
 
class MyStrategy(strategy.BacktestingStrategy):
	def __init__(self, feed):
		super().__init__(feed, 1000000)
		
		self.setUseAdjustedValues(True)
		
		order = {
			"ibm" : 1996,
			"aes" : 22565,
			"aig" : 5445,
			"orcl" : 8582
			}
		for instrument, quantity in order.items():
			# self.marketOrder(instrument, quantity, onClose = True, allOrNone = True)
			self.enterLong(instrument, quantity, True, True)
			
	def onEnterOk(self, position):
		execInfo = position.getEntryOrder().getExecutionInfo()
		print("BUY at $%.2f 数量:%d 手续费:%.2f" % (execInfo.getPrice(), execInfo.getQuantity(), execInfo.getCommission()))
		
	def onBars(self, bars):
		pass
			
		
if __name__ == "__main__":
	feed = quandlfeed.Feed()
	feed.addBarsFromCSV("ibm", "WIKI-IBM-2011-quandl.csv")
	feed.addBarsFromCSV("aes", "WIKI-AES-2011-quandl.csv")
	feed.addBarsFromCSV("aig", "WIKI-AIG-2011-quandl.csv")
	feed.addBarsFromCSV("orcl", "WIKI-ORCL-2011-quandl.csv")
	
	myStrategy = MyStrategy(feed)

	retAnalyzer = returns.Returns()
	myStrategy.attachAnalyzer(retAnalyzer)
	sharpeAnalyzer = sharpe.SharpeRatio()
	myStrategy.attachAnalyzer(sharpeAnalyzer)
	
	myStrategy.run()
	
	print("最终收益:%.2f" % myStrategy.getResult())
	print("年化收益率:%.2f%%" % (retAnalyzer.getCumulativeReturns()[-1]*100))
	print("平均每日收益率:%.2f %%" % (stats.mean(retAnalyzer.getReturns())*100))
	print("每日收益标准差:%.4f" % (stats.stddev(retAnalyzer.getReturns())))
	print("夏普比率:%.2f" % sharpeAnalyzer.getSharpeRatio(0))
 