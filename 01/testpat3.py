from pyalgotrade import plotter, strategy
from pyalgotrade.stratanalyzer import sharpe
from pyalgotrade.technical import ma
from pyalgotrade_tushare import tools
from pandas.plotting import register_matplotlib_converters
import tsfeed
from GetHistroyData import *


class MyStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument):
        super().__init__(feed)

        self.__position = None
        self.__sma = ma.SMA(feed[instrument].getCloseDataSeries(), 150)
        self.__instrument = instrument
        self.getBroker()

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

    def getSMA(self):
        return self.__sma

    def onBars(self, bars):
        # 每一个数据都会抵达这里，就像becktest中的next
        # Wait for enough bars to be available to calculate a SMA.
        if self.__sma[-1] is None:
            return

        # bar.getTyoicalPrice = (bar.getHigh() + bar.getLow() + bar.getClose())/ 3.0
        bar = bars[self.__instrument]

        # If a position was not opened, check if we should enter a long position.
        if self.__position is None:
            if bar.getPrice() > self.__sma[-1]:
                # 开多头.
                self.__position = self.enterLong(self.__instrument, 100, True)

        # 平掉多头头寸.
        elif bar.getPrice() < self.__sma[-1] and not self.__position.exitActive():
            self.__position.exitMarket()


if __name__ == '__main__':
    register_matplotlib_converters()
    instruments = ["600036"]
    # feeds = tools.build_feed(instruments, 2013, 2016, "histdata")
    
    data = GetHistroyData("600036", "2013-01-01", "2017-01-01")
    data.to_csv("600036.csv")
    
    feeds = tsfeed.Feed()
    feeds.addBarsFromCsv("600036", "600036.csv", "2013-01-01", "2017-01-01")

    # 3.实例化策略
    strat = MyStrategy(feeds, instruments[0])

    # 4.设置指标和绘图
    ratio = sharpe.SharpeRatio()
    strat.attachAnalyzer(ratio)
    plter = plotter.StrategyPlotter(strat)

    # 5.运行策略
    strat.run()
    strat.info("最终收益: %.2f" % strat.getResult())

    # 6.输出夏普率、绘图
    strat.info("夏普比率: " + str(ratio.getSharpeRatio(0)))
    plter.savePlot("600036.png")
    