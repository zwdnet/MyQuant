# coding:utf-8
# 用backtrader对定投实盘记录进行回测


import backtrader as bt
import backtrader.analyzers as btay
import tushare as ts
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt


# 获取数据
def getData(code, start, end):
    filename = code+".csv"
    print("./" + filename)
    # 已有数据文件，直接读取数据
    if os.path.exists("./" + filename):
        df = pd.read_csv(filename)
    else: # 没有数据文件，用tushare下载
        df = ts.get_k_data(code, autype = "qfq", start = start,  end = end)
        df.to_csv(filename)
    df.index = pd.to_datetime(df.date)
    df['openinterest']=0
    df=df[['open','high','low','close','volume','openinterest']]
    return df
    
    
# 建立数据源
def createDataFeeds(code, name, start, end):

    df_data = getData(code, start, end)
    # print(df_300.info(), df_nas.info())
    # 建立数据源
    start_date = list(map(int, start.split("-")))
    end_date = list(map(int, end.split("-")))
    data = bt.feeds.PandasData(dataname = df_data, name = name, fromdate = datetime.datetime(start_date[0], start_date[1], start_date[2]), todate = datetime.datetime(end_date[0], end_date[1], end_date[2]))
    return data
    
    
# 建立回测模型
def createBacktesting(commis = 0.0003, initcash = 0.01):
    # 建立回测实例，加载数据，策略。
    cerebro = bt.Cerebro()
    cerebro.addstrategy(TradeStrategy)
    # 添加回撤观察器
    cerebro.addobserver(bt.observers.DrawDown)
    # 设置手续费
    cerebro.broker.setcommission(commission=commis)
    # 设置初始资金为0.01
    cerebro.broker.setcash(initcash)
    # 添加分析对象
    cerebro.addanalyzer(btay.SharpeRatio, _name = "sharpe", riskfreerate = 0.02)
    cerebro.addanalyzer(btay.AnnualReturn, _name = "AR")
    cerebro.addanalyzer(btay.DrawDown, _name = "DD")
    cerebro.addanalyzer(btay.Returns, _name = "RE")
    cerebro.addanalyzer(btay.TradeAnalyzer, _name = "TA")
    return cerebro
    
    
# 添加数据源
def addData(cerebro, dataFeed, name):
    cerebro.adddata(dataFeed, name = name)
    
# 输出结果
def outputResult(cerebro):
    cerebro.plot(numfigs = 2)
    plt.savefig("result.png")
    print("夏普比例:", results[0].analyzers.sharpe.get_analysis()["sharperatio"])
    print("年化收益率:", results[0].analyzers.AR.get_analysis())
    print("最大回撤:%.2f，最大回撤周期%d" % (results[0].analyzers.DD.get_analysis().max.drawdown, results[0].analyzers.DD.get_analysis().max.len))
    print("总收益率:%.2f" % (results[0].analyzers.RE.get_analysis()["rtot"]))
    results[0].analyzers.TA.print()
    
    
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
    name300 = "300ETF"
    nameNas = "nasETF"
    data300 = createDataFeeds("510300", name300, start, end)
    dataNas = createDataFeeds("513100", nameNas, start, end)
    cerebro = createBacktesting()
    addData(cerebro, data300, name300)
    addData(cerebro, dataNas, nameNas)
    # 运行回测
    print("初始资金:%.2f" % cerebro.broker.getvalue())
    results = cerebro.run()
    print("期末资金:%.2f" % cerebro.broker.getvalue())
    outputResult(cerebro)
    
    