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

    def onEnterOk(self, position):
        execInfo = position.getEntryOrder().getExecutionInfo()
        # self.info("买入 %.2f" % (execInfo.getPrice()))

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
    	brk = self.getBroker()
    	shares = 100
    	# 策略买入
    	for inst in self.__instrument:
    		price = bars[inst].getPrice()
    		if brk.getCash() < price*shares:
    			self.info("现金不足")
    			return
    		self.__position = self.enterLong(inst, shares, True)
    		self.__cost += brk.getCommission().calculate(brk, price, shares)
    		# self.info("股票代码%s 可用现金%.2f 股价%.2f 持股数量%d 市值1:%.2f 市值2:%.2f 交易成本%.2f" % (inst, brk.getCash(), price, brk.getShares(inst), brk.getEquity(), self.getResult(), self.__cost))
    	# x = input("按任意键继续")


if __name__ == '__main__':
	instruments = ["600519", "601398", "601318"]
	bt = Backtesting(["600519"], 2014, 2018, MyStrategy, cash = 10000000)
	strategy = bt.getStrategy()
	strategy[0].run()
	strategy[1].run()
	result = bt.getResult()
	bt.outputResult()
	print("期末总资产%.2f" % strategy[0].getResult())
	bt.drawResult("results.png")
    