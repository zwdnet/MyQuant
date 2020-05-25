# coding:utf-8
# 实现R_Breaker策略


from pyalgotrade import plotter, strategy
from pyalgotrade.stratanalyzer import sharpe
from pandas.plotting import register_matplotlib_converters
from getData import downloadData, buildFeed
from pyalgotrade.stratanalyzer import returns, sharpe, drawdown, trades
from pyalgotrade import broker


# 策略要计算的指标
class Index():
    def __init__(self):
        self.__pivot = 0.0     # 枢轴点
        self.__bBreak = 0.0 # 突破买入价
        self.__sSetup = 0.0 # 观察卖出价
        self.__sEnter = 0.0  # 反转卖出价
        self.__bEnter = 0.0  # 反转买入价
        self.__bSetup = 0.0 # 观察买入价
        self.__sBreak = 0.0 # 突破卖出价
        
        
   # 根据前一日最高价，最低价，收盘价更新指标
    def updata(self, high, low, close):
        self.__pivot = (high+low+close)/3.0
        self.__bBreak = high+2.0*(self.__pivot-low)
        self.__sSetup = self.__pivot+(high-low)
        self.__sEnter = 2*self.__pivot-low
        self.__bEnter = 2*self.__pivot-high
        self.__bSetup = self.__pivot-(high-low)
        self.__sBreak = low-2*(high-self.__pivot)
       
       
    # 返回所有指标
    def getIndex(self):
        return (self.__bBreak, self.__sSetup, self.__sEnter, self.__bEnter, self.__bSetup, self.__sBreak)
        
        
    # 根据当前的价格，判断是否操作
    # 返回值，0-不操作，1-全仓买入，2-全仓卖出
    def judge(self, high, low, close, share, price):
        # 空仓且价格超过突破买入价，全仓买入
        if share == 0 and price > self.__bBreak:
            return 1
        # 满仓，日内最高价超过观察卖出价后，盘中价格回落跌破反转卖出价，清仓。
        if share != 0 and high > self.__sSetup and price < self.__sEnter:
            return 2
        # 其它情况，不操作
        return 0
        
       

# 策略类
class MyStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument, brk):
        super().__init__(feed, brk)
        self.__position = None
        self.__instrument = instrument
        self.getBroker()
        self.__cost = 0.0
        # 记录当前的日期
        self.__year = 0
        self.__month = 0
        self.__day = 0
        # 策略的指标，每天更新
        self.__index = Index()
        # 每天的最高价，最低价，收盘价
        self.__high = 0
        self.__low = 100000000
        self.__close = 0

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
        
    # 日期改变，重新计算那六个指标
    def updateData(self):
        self.__index.updata(self.__high, self.__low, self.__close)
        self.__high = 0
        self.__low = 100000000
        self.__close = 0

    def onBars(self, bars):
        brk = self.getBroker()
        bar = bars[self.__instrument]
        # 先判断是否是新的交易日，是则更新指标
        date = bar.getDateTime()
        year = date.year
        month = date.month
        day = date.day
        print(date, self.__high, self.__low, self.__close)
        if self.__year != year or self.__month != month or self.__day != day:
            self.__year = year
            self.__month = month
            self.__day = day
            self.updateData()
        else:
            price = bars[self.__instrument].getPrice()
            if price > self.__high:
                self.__high = price
            if price < self.__low:
                self.__low = price
            self.__close = price
        share = brk.getShares(self.__instrument)
        price = bars[self.__instrument].getPrice()
        tradeCode = self.__index.judge(self.__high, self.__low, self.__close, share, price)
        if tradeCode == 1:
            # 全仓买入
            if shares != 0:
                break
            else: #这里全仓买入
                cash = brk.getCash()
                shares = cash/price
                shares = (shares//100)*100
                self.__position = self.enterLong(self.__instrument, shares, True)
                self.__cost += brk.getCommission().calculate(brk, price, shares)
        elif tradeCode == 2:
            # 全仓卖出
            if share == 0:
                break
            else: #这里全仓卖出
                self.__position.exitMarket()
                self.__cost += brk.getCommission().calculate(brk, price, shares)
        
            
        #shares = 100
        #price = bars[self.__instrument].getPrice()
        #if brk.getCash() < price*shares:
        #    self.info("现金不足")
        #    return
        #self.__position = self.enterLong(self.__instrument, shares, True)
        #self.__cost += brk.getCommission().calculate(brk, price, shares)
        #self.info("可用现金%.2f 股价%.2f 持股数量%d 市值1:%.2f 市值2:%.2f 计算市值:%.2f 交易成本%.2f" % (brk.getCash(), price, brk.getShares(self.__instrument), brk.getEquity(), self.getResult(), (brk.getCash() + brk.getShares(self.__instrument)*price), self.__cost))
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
#    instruments = ["000001"]
#    feeds = tools.build_feed(instruments, 2016, 2018, "histdata")
    
    instruments = ["601988"]
    feeds = buildFeed(instruments[0],downloadData(instruments[0]))
    
    # 设置手续费, 万分之一
    broker_commision = broker.backtesting.TradePercentage(0.0001)
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
    # analyzer(result[1:5])
    
    res = result[0].getResult()
    print("期末总资产%.2f 期末收益率%.2f%%" % (res, 100.0*(res/cash-1.0)))