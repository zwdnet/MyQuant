# coding:utf-8
# 封装回测的过程


from pyalgotrade import strategy, broker
from pyalgotrade_tushare import tools, barfeed


"""
封装回测过程
参数:
	instrument: 要回测的股票代码
	startYear: 回测开始年份
	endYear: 回测结束年份
	strategy: 回测的策略
	base: 基准股票代码，默认为300etf
	cash: 初始资金，默认为1000000元
	feeRate: 手续费费率，默认为0.0003
"""
class Backtesting():
	def __init__(self, instrument, startYear, endYear, strategy, base = "510300", cash = 1000000, feeRate = 0.0003):
		self.__instrument = instrument
		self.__startYear = startYear
		self.__endYear = endYear
		self.__strategy = strategy
		self.__base = base
		self.__cash = cash
		self.__feeRate = feeRate
		# 要创建的内部变量
		self.__strategyTest = None
		self.__feed = None
		self.__strategyBase = None
		self.__feedBase = None
		self.__brk = None
		self.__brkBase = None
		
	# 创建barfeed数据源
	def createBarfeed(self):
		self.__feed = tools.build_feed(self.__instruments, self.__startYear, self.__endYear, "histdata")
		self.__feedBase = tools.build_feed(self.__base, self.__startYear, self.__endYear, "histdata")
		
	# 创建broker
	def createBroker(self):
		# 设置手续费
		broker_commision = broker.backtesting.TradePercentage(self.__feeRate)
		self.__brk = broker.backtesting.Broker(self.__cash, self.__feed, broker_commision)
		self.__brk = broker.backtesting.Broker(self.__cash, self.__feedBase, broker_commision)
		
	# 创建策略
	def createStrategy(self):
		self.__strategyTest = self.__strategy(self.__feed, self.__instrument, self.__brk)
		self.__strategyBase = self.__strategy(self.__feedBase, self.__base, self.__brk)
    	
		
if __name__ == "__main__":
	pass
	