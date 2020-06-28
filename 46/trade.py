# coding:utf-8
# 用backtrader对定投实盘记录进行回测


import backtrader as bt
import tushare as ts
import os
import pandas as pd
import datetime


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
    
    
# 交易策略
class TradeStrategy(bt.Strategy):
    params = (
            ("recordFilename", "etfdata.csv"),
    )
    
    def __init__(self):
        df_record = pd.read_csv(self.params.recordFilename)
        df_record.index = pd.to_datetime(df_record.成交日期, format = "%Y%m%d")
        df_record.drop(labels = "成交日期", axis = 1, inplace = True)
        print(df_record.head())
        
        
    def next(self):
        for data in self.datas:
            print(data.datetime.date(0))
            print("name :%s, price:%.2f" % (data._name, data[0]))


if __name__ == "__main__":
    start = "2018-01-01"
    end = "2020-05-31"
    df_300 = getData("510300", start, end)
    df_nas = getData("513100", start, end)
    print(df_300.info(), df_nas.info())
    # 建立数据源
    start_date = list(map(int, start.split("-")))
    end_date = list(map(int, end.split("-")))
    data300 = bt.feeds.PandasData(dataname = df_300, name = "300ETF", fromdate = datetime.datetime(start_date[0], start_date[1], start_date[2]), todate = datetime.datetime(end_date[0], end_date[1], end_date[2]))
    dataNas = bt.feeds.PandasData(dataname = df_nas, name = "纳指ETF", fromdate = datetime.datetime(start_date[0], start_date[1], start_date[2]), todate = datetime.datetime(end_date[0], end_date[1], end_date[2]))
    # 建立回测实例，加载数据，策略。
    cerebro = bt.Cerebro()
    cerebro.addstrategy(TradeStrategy)
    cerebro.adddata(data300, name = "300ETF")
    cerebro.adddata(dataNas, name = "纳指ETF")
    # 运行回测
    cerebro.run()
    