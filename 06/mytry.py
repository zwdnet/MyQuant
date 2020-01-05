# coding:utf-8
from pyalgotrade import strategy
from pyalgotrade import plotter
from pyalgotrade.tools import quandl
from pyalgotrade.technical import vwap
from pyalgotrade.stratanalyzer import sharpe


class VWAPMomentum(strategy.BacktestingStrategy):
	def __init__(self, feed, instrument, vwapWindowSize, threshold):
		super().__init__(feed)
		self.__instrument = instrument
		self.__vwap = vwap.VWAP(feed[instrument], vwapWindowSize)
		self.__threshold = threshold
		self.__position = None
		
	def getVWAP(self):
		return self.__vwap
		
	def onEnterOk(self, position):
		execInfo = position.getEntryOrder().getExecutionInfo()
		self.info("BUY at $%.2f 数量:%d 手续费:%.2f" % (execInfo.getPrice(), execInfo.getQuantity(), execInfo.getCommission()))
		
	def onEnterCanceled(self, position):
		self.__position = None
		self.info("买入失败")
		
	def onExitOk(self, position):
		execInfo = position.getExitOrder().getExecutionInfo()
		self.info("SELL at $%.2f" % (execInfo.getPrice()))
		self.__position = None
		
	def onExitCanceled(self, position):
		self.info("卖出失败")
		self.enterShort(self.__instrument, 100, True)
		
	def onBars(self, bars):
		vwap = self.__vwap[-1]
		if vwap == None:
			return
			
		shares = self.getBroker().getShares(self.__instrument)
		price = bars[self.__instrument].getClose()
		notional = shares * price
		
		if price > vwap * (1 + self.__threshold) and notional < 1000000:
			#self.marketOrder(self.__instrument, 100)
			self.enterLong(self.__instrument, 100, True)
		elif price < vwap * (1 - self.__threshold) and notional > 0:
			#self.marketOrder(self.__instrument, -100)
			self.enterShort(self.__instrument, 100, True)
		brk = self.getBroker()
		print("剩余现金%.2f" % brk.getCash())
			
	
def main():
	instrument = "AAPL"
	vwapWindowSize = 5
	threshold = 0.01
	
	feed = quandl.build_feed("WIKI", [instrument], 2011, 2012, ".")
	
	strat = VWAPMomentum(feed, instrument, vwapWindowSize, threshold)
	sharpeRatioAnalyzer = sharpe.SharpeRatio()
	strat.attachAnalyzer(sharpeRatioAnalyzer)
	
	plter = plotter.StrategyPlotter(strat, True, True, True)
	plter.getInstrumentSubplot(instrument).addDataSeries("vwap", strat.getVWAP())
		
	strat.run()
	print("夏普比例:%.2f" % sharpeRatioAnalyzer.getSharpeRatio(0.03))
	print("期末策略市值:%.2f" % strat.getResult())
	
	plter.savePlot("mytry.png")
		
if __name__ == "__main__":
	main()
		