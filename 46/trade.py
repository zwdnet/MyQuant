# coding:utf-8
# 用backtrader对定投实盘记录进行回测


import backtrader as bt
import backtest
import pandas as pd


# 交易策略
class TradeStrategy(bt.Strategy):
    params = (
            ("recordFilename", "etfdata.csv"),
            ("printlog", False)
    )
    
    def __init__(self):
        self.df_record = pd.read_csv(self.params.recordFilename)
        self.df_record.成交日期 = pd.to_datetime(self.df_record.成交日期, format = "%Y%m%d")
        self.df_record.index = self.df_record.成交日期
        self.df_record.drop(labels = "成交日期", axis = 1, inplace = True)
        # print(self.df_record.head(), self.df_record.info())
        self.order = None
        ad = bt.indicators.AroonDown(plotname = "AD")
        ad.plotinfo.subplot = True
        
        
    def log(self, txt, dt=None, doprint=False):
        '''log记录'''
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))
            
    # 输出交易过程
    def __displayOrder(self, buy, order):
        if buy:
            self.log(
                    '执行买入, 价格: %.2f, 成本: %.2f, 手续费 %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))
        else:
            self.log(
                    '执行卖出, 价格: %.2f, 成本: %.2f, 手续费 %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))
            
    def notify_order(self, order):
        # 有交易提交/被接受，啥也不做
        if order.status in [order.Submitted, order.Accepted]:
            return

        # 检查一个交易是否完成。
        # 如果钱不够，交易会被拒绝。
        if order.status in [order.Completed]:
            if order.isbuy():
                self.__displayOrder(True, order)
            elif order.issell():
                self.__displayOrder(False, order)

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('交易取消/被拒绝。')

        self.order = None
        
    # 交易的工具函数
    def __doTrade(self, data, name, price, stock, commit, orderType):
        if stock > 0 and name == data._name:
            self.broker.add_cash(price*stock + commit)
            self.order = self.buy(data = data, size = stock, price = price, exectype = orderType)
        elif stock < 0 and name == data._name:
            self.order = self.sell(data = data, size = -1*stock, price = price, exectype = orderType)
            
    # 具体交易逻辑，可以改的。
    def doTrade(self):
        tradeData = pd.DataFrame()
        orderType = bt.Order.Market
        for data in self.datas:
            date = data.datetime.date(0)
            tradeBar = self.df_record.loc[date.strftime("%Y-%m-%d"),:]
            # print("bar数据", date, data._name)
            if len(tradeBar) != 0:
                for i in range(len(tradeBar)):
                    name = tradeBar.iloc[i].证券名称 
                    price = tradeBar.iloc[i].成交均价 
                    stock = tradeBar.iloc[i].成交量 
                    commit = tradeBar.iloc[i].手续费
                    # 进行交易
                    self.__doTrade(data, name, price, stock, commit, orderType)
            
    def next(self):
        if self.order:
            return
        self.doTrade()
                        
    def stop(self):
        self.log("最大回撤:-%.2f%%" % self.stats.drawdown.maxdrawdown[-1], doprint=True)
                    


if __name__ == "__main__":
    # 加载数据，建立数据源
    start = "2018-01-01"
    end = "2020-07-05"
    name = ["300ETF", "nasETF"]
    code = ["510300", "513100"]
    backtest = backtest.BackTest(TradeStrategy, start, end, code, name)
    backtest.run()
    backtest.output()
