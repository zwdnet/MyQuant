# coding:utf-8
# 封装回测的过程


from pyalgotrade import strategy, broker, plotter
from pyalgotrade_tushare import tools, barfeed
from pyalgotrade.stratanalyzer import returns, sharpe, drawdown, trades
import pandas as pd
from scipy import stats
import numpy as np
from pandas.plotting import register_matplotlib_converters


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
		self.__base = []
		self.__base.append(base)
		self.__cash = cash
		self.__feeRate = feeRate
		# 要创建的内部变量
		self.__strategyTest = None
		self.__feed = None
		self.__strategyBase = None
		self.__feedBase = None
		self.__brk = None
		self.__brkBase = None
		self.__return = returns.Returns()
		self.__returnBase = returns.Returns()
		self.__sharpe = sharpe.SharpeRatio()
		self.__drawdown = drawdown.DrawDown()
		self.__trade = trades.Trades()
		self.__result = pd.DataFrame()
		self.__plter = None
		# 使用pyalgotrade绘图要加上，不然报错
		register_matplotlib_converters()
		
	# 创建barfeed数据源
	def __createBarfeed(self):
		self.__feed = tools.build_feed(self.__instrument, self.__startYear, self.__endYear, "histdata")
		self.__feedBase = tools.build_feed(self.__base, self.__startYear, self.__endYear, "histdata")
		
	# 创建broker
	def __createBroker(self):
		# 设置手续费
		broker_commision = broker.backtesting.TradePercentage(self.__feeRate)
		self.__brk = broker.backtesting.Broker(self.__cash, self.__feed, broker_commision)
		self.__brkBase = broker.backtesting.Broker(self.__cash, self.__feedBase, broker_commision)
		
	# 创建策略并绑定分析器
	def __createStrategy(self):
		self.__strategyTest = self.__strategy(self.__feed, self.__instrument, self.__brk)
		self.__strategyTest.attachAnalyzer(self.__return)
		self.__strategyTest.attachAnalyzer(self.__sharpe)
		self.__strategyTest.attachAnalyzer(self.__drawdown)
		self.__strategyTest.attachAnalyzer(self.__trade)
		self.__strategyBase = self.__strategy(self.__feedBase, self.__base[0], self.__brkBase)
		self.__strategyBase.attachAnalyzer(self.__returnBase)
		
	# 创建绘图器
	def __createPlter(self):
		self.__plter = plotter.StrategyPlotter(self.__strategyTest)
		# self.__plter.getOrCreateSubplot("return").addDataSeries("retuens", self.__return.getReturns())
		# self.__plter.getOrCreateSubplot("CumReturn").addDataSeries("CumReturn", self.__return.getCumulativeReturns())
		# self.__plter.getOrCreateSubplot("return").addDataSeries("retuensBase", self.__returnBase.getReturns())
		# self.__plter.getOrCreateSubplot("CumReturn").addDataSeries("CumReturnBase", self.__returnBase.getCumulativeReturns())
		
		
	# 计算αβ信息比例等指标
	def __alphaBeta(self):
		# 计算α β值
		X = self.__return.getCumulativeReturns()
		Y = self.__returnBase.getCumulativeReturns()
		n1 = X.__len__()
		n2 = Y.__len__()
		x = []
		y = []
		if n1 == n2:
			for i in range(n1):
				x.append(X[i])
				y.append(Y[i])
		alpha = 0.0
		beta = 0.0
		b, a, r_value, p_value, std_err = stats.linregress(x, y)
		# alpha转化为年
		alpha = [round(a * 250, 3)]
		beta = [round(b, 3)]
		self.__result["alpha"] = alpha
		self.__result["beta"] = beta
	
		# 计算信息比率
		# 先计算超额收益
		ex_return = [x[i] - y[i] for i in range(len(x))]
		information = (x[-1] - y[-1])/np.std(ex_return)
		self.__result["信息比率"] = information
		
	# 计算回测指标
	def __testResults(self):
		# 计算年化收益率
		self.__result["总收益率"] = [self.__return.getCumulativeReturns()[-1]]
		# 计算夏普比率
		self.__result["夏普比率"] = [self.__sharpe.getSharpeRatio(0.05)]
		# 计算最大回撤
		self.__result["最大回撤"] = [self.__drawdown.getMaxDrawDown()]
		self.__result["最大回撤期间"] = [self.__drawdown.getLongestDrawDownDuration()]
		self.__alphaBeta()
		
	# 获得回测结果
	def getResult(self):
		self.__testResults()
		return self.__result
		
	# 建立策略并返回
	def getStrategy(self):
		self.__createBarfeed()
		self.__createBroker()
		self.__createStrategy()
		self.__createPlter()
		return (self.__strategyTest, self.__strategyBase)
		
	# 输出回测结果指标
	def outputResult(self):
		print("总收益率:%.2f" % self.__result["总收益率"])
		print("夏普比率:%.2f" % self.__result["夏普比率"])
		print("最大回撤:%.2f" % self.__result["最大回撤"])
		print("最大回撤期间:%s" % self.__result["最大回撤期间"])
		print("alpha:%.2f" % self.__result["alpha"])
		print("beta:%.2f" % self.__result["beta"])
		print("信息比率:%.2f" % self.__result["信息比率"])
		
	# 绘图输出回测结果
	def drawResult(self, filename):
		self.__plter.savePlot(filename)
		
		
if __name__ == "__main__":
	pass
	