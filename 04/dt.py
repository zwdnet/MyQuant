# coding:utf-8
# 回测定投测量


from pyalgotrade import strategy
from pyalgotrade import broker
from scipy import stats
import numpy as np
from Backtesting import Backtesting


class MyStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument, brk):
        super().__init__(feed, brk)
        self.__position = None
        self.__instrument = instrument
        self.getBroker()
        self.__cost = 0.0
        self.__t = 0

    def onEnterOk(self, position):
        execInfo = position.getEntryOrder().getExecutionInfo()
        self.info("买入 %.2f" % (execInfo.getPrice()))

    def onEnterCanceled(self, position):
        self.__position = None

    def onExitOk(self, position):
        execInfo = position.getExitOrder().getExecutionInfo()
        self.info("卖出 %.2f" % (execInfo.getPrice()))
        self.__position = None

    def onExitCanceled(self, position):
        # If the exit was canceled, re-submit it.
        self.__position.exitMarket()
        

    def onBars(self, bars):
    	# 每八个交易日交易一次
    	if self.__t < 8:
    		self.__t += 1
    		return
    	self.__t = 0
    	brk = self.getBroker()
    	shares = 100
    	# 策略买入
    	if len(self.__instrument) == 2:
    		for inst in self.__instrument:
    			price = bars[inst].getPrice()
    			if brk.getCash() < price*shares:
    				self.info("现金不足")
    				return
    			self.__position = self.enterLong(inst, shares, True)
    			self.__cost += brk.getCommission().calculate(brk, price, shares)
    			# self.info("可用现金%.2f 股价%.2f 持股数量%d 市值1:%.2f 市值2:%.2f 交易成本%.2f" % (brk.getCash(), price, brk.getShares(inst), brk.getEquity(), self.getResult(), self.__cost))
    	# 回测基准买入
    	else:
    		price = bars[self.__instrument].getPrice()
    		if brk.getCash() < price*shares:
    			self.info("现金不足")
    			return
    		self.__position = self.enterLong(self.__instrument, shares, True)
    		self.__cost += brk.getCommission().calculate(brk, price, shares)
    	# x = input("按任意键继续")


def analyzer(testResult):
	# 回测信息
	print("年化收益率: %.2f %%" % (testResult[0].getCumulativeReturns()[-1] * 100))
	print("夏普比率: %.2f" % (testResult[1].getSharpeRatio(0.05)))
	print("最大回撤: %.2f %%" % (testResult[2].getMaxDrawDown() * 100))
	print("最大回撤期间: %s" % (testResult[2].getLongestDrawDownDuration()))
	
	# 交易信息
	td = testResult[3]
	print("-----------------------")
	print("总交易次数:%d" % (td.getCount()))
	if td.getCount() > 0:
		profits = td.getAll()
		print("平均收益:%.2f" % (profits.mean()))
		print("收益标准差:%.2f" % (profits.std()))
		print("最大收益:%.2f" % (profits.max()))
		print("最小收益:%.2f" % (profits.min()))
		returns = td.getAllReturns()
		print("平均收益率:%.2f%%" % (returns.mean() * 100))
		print("收益率标准差:%.2f%%" % (returns.std() * 100))
		print("最大收益率:%.2f%%" % (returns.max() * 100))
		print("最小收益率:%.2f%%" % (returns.min() * 100))
	
	print("-----------------------")	
	print("盈利的交易次数: %d" % (td.getProfitableCount()))
	if td.getProfitableCount() > 0:
		profits = td.getProfits()
		print("平均收益:%.2f" % (profits.mean()))
		print("收益标准差:%.2f" % (profits.std()))
		print("最大收益:%.2f" % (profits.max()))
		print("最小收益:%.2f" % (profits.min()))
		returns = td.getPositiveReturns()
		print("平均收益率:%.2f%%" % (returns.mean() * 100))
		print("收益率标准差:%.2f%%" % (returns.std() * 100))
		print("最大收益率:%.2f%%" % (returns.max() * 100))
		print("最小收益率:%.2f%%" % (returns.min() * 100))
		
	print("-----------------------")
	print("未盈利的交易次数: %d" % (td.getUnprofitableCount()))
	if td.getUnprofitableCount() > 0:
		losses = td.getLosses()
		print("平均收益:%.2f" % (losses.mean()))
		print("收益标准差:%.2f" % (losses.std()))
		print("最大收益:%.2f" % (losses.max()))
		print("最小收益:%.2f" % (losses.min()))
		returns = td.getNegativeReturns()
		print("平均收益率:%.2f%%" % (returns.mean() * 100))
		print("收益率标准差:%.2f%%" % (returns.std() * 100))
		print("最大收益率:%.2f%%" % (returns.max() * 100))
		print("最小收益率:%.2f%%" % (returns.min() * 100))


if __name__ == '__main__':
    bt = Backtesting(["510300", "513100"], 2018, 2019, MyStrategy, cash = 100000)
    strategy = bt.getStrategy()
    strategy[0].run()
    strategy[1].run()
    result = bt.getResult()
    bt.outputResult()
    print("期末总资产%.2f" % strategy[0].getResult())
    bt.drawResult("test.png")
    