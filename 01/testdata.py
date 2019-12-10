from pyalgotrade import plotter, strategy
from pyalgotrade.stratanalyzer import sharpe
from pandas.plotting import register_matplotlib_converters
from pyalgotrade_tushare import tools, barfeed
from pyalgotrade.stratanalyzer import returns, sharpe, drawdown, trades
from pyalgotrade import broker


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
    	price = bars[self.__instrument].getPrice()
    	if brk.getCash() < price*shares:
    		self.info("现金不足")
    		return
    	self.__position = self.enterLong(self.__instrument, shares, True)
    	self.__cost += brk.getCommission().calculate(brk, price, shares)
    	self.info("可用现金%.2f 股价%.2f 持股数量%d 市值1:%.2f 市值2:%.2f 计算市值:%.2f 交易成本%.2f" % (brk.getCash(), price, brk.getShares(self.__instrument), brk.getEquity(), self.getResult(), (brk.getCash() + brk.getShares(self.__instrument)*price), self.__cost))
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


def run_strategy(cash):
    instruments = ["000001"]
    feeds = tools.build_feed(instruments, 2016, 2018, "histdata")
    
    # 设置手续费
    broker_commision = broker.backtesting.TradePercentage(0.0003)
    brk = broker.backtesting.Broker(cash, feeds, broker_commision)
	
    myStrategy = MyStrategy(feeds, instruments[0], brk)
    retAnalyzer = returns.Returns()
    myStrategy.attachAnalyzer(retAnalyzer)
    sharpeAnalyzer = sharpe.SharpeRatio()
    myStrategy.attachAnalyzer(sharpeAnalyzer)
    drawDownAnalyzer = drawdown.DrawDown()
    myStrategy.attachAnalyzer(drawDownAnalyzer)
    tradesAnalyzer = trades.Trades()
    myStrategy.attachAnalyzer(tradesAnalyzer)
    
    plter = plotter.StrategyPlotter(myStrategy)
    plter.getOrCreateSubplot("return").addDataSeries("retuens", retAnalyzer.getReturns())
    plter.getOrCreateSubplot("CumReturn").addDataSeries("CumReturn", retAnalyzer.getCumulativeReturns())
    
    
    myStrategy.run()
    plter.savePlot("testdata.png")
    print("交易次数:%d" % (tradesAnalyzer.getCount()))
    return (myStrategy, retAnalyzer, sharpeAnalyzer, drawDownAnalyzer, tradesAnalyzer)


if __name__ == '__main__':
    register_matplotlib_converters()
    cash = 1000000
    result = run_strategy(cash)
    analyzer(result[1:5])
    
    res = result[0].getResult()
    print("期末总资产%.2f 期末收益率%.2f%%" % (res, 100.0*(res/cash-1.0)))