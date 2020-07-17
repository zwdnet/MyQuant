# coding:utf-8
# 量化交易回测类


import backtrader as bt
import backtrader.analyzers as btay
import tushare as ts
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt


# 回测类
class BackTest:
    def __init__(self, strategy, start, end, code, name):
        self.__cerebro = None
        self.__strategy = strategy
        self.__start = start
        self.__end = end
        self.__code = code
        self.__name = name
        self.__result = None
        self.__commission = 0.0003
        self.__initcash = 0.01
        self.init()
        
    # 真正进行初始化的地方
    def init(self):
        self.__cerebro = bt.Cerebro()
        self.__cerebro.addstrategy(self.__strategy)
        self.settingCerebro()
        self.createDataFeeds()
        
    # 设置cerebro
    def settingCerebro(self):
        # 添加回撤观察器
        self.__cerebro.addobserver(bt.observers.DrawDown)
        # 设置手续费
        self.__cerebro.broker.setcommission(commission=self.__commission)
        # 设置初始资金为0.01
        self.__cerebro.broker.setcash(self.__initcash)
        # 添加分析对象
        self.__cerebro.addanalyzer(btay.SharpeRatio, _name = "sharpe", riskfreerate = 0.02)
        self.__cerebro.addanalyzer(btay.AnnualReturn, _name = "AR")
        self.__cerebro.addanalyzer(btay.DrawDown, _name = "DD")
        self.__cerebro.addanalyzer(btay.Returns, _name = "RE")
        self.__cerebro.addanalyzer(btay.TradeAnalyzer, _name = "TA")
        
    # 建立数据源
    def createDataFeeds(self):
        for i in range(len(self.__code)):
            df_data = self._getData(self.__code[i])
            start_date = list(map(int, self.__start.split("-")))
            end_date = list(map(int, self.__end.split("-")))
            dataFeed = bt.feeds.PandasData(dataname = df_data, name = self.__name[i], fromdate = datetime.datetime(start_date[0], start_date[1], start_date[2]), todate = datetime.datetime(end_date[0], end_date[1], end_date[2]))
            self.__cerebro.adddata(dataFeed, name = self.__name[i])
            
    # 获取账户总价值
    def getValue(self):
        return self.__cerebro.broker.getvalue()
        
    # 执行回测
    def run(self):
        print("初始资金:%.2f" % self.getValue())
        self.__results = self.__cerebro.run()
        print("期末资金:%.2f" % self.getValue())
        
    # 输出回测结果
    def output(self):
        self.__cerebro.plot(numfigs = 2)
        plt.savefig("result.png")
        print("夏普比例:", self.__results[0].analyzers.sharpe.get_analysis()["sharperatio"])
        print("年化收益率:", self.__results[0].analyzers.AR.get_analysis())
        print("最大回撤:%.2f，最大回撤周期%d" % (self.__results[0].analyzers.DD.get_analysis().max.drawdown, self.__results[0].analyzers.DD.get_analysis().max.len))
        print("总收益率:%.2f" % (self.__results[0].analyzers.RE.get_analysis()["rtot"]))
        self.__results[0].analyzers.TA.print()
            
    # 获取数据
    def _getData(self, code):
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
