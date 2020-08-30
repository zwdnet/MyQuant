# coding:utf-8
# 测试多参数优化


import backtrader as bt
import pandas as pd
import tushare as ts
import os
import datetime
import backtrader.analyzers as btay


class TestStrategy(bt.Strategy):
    params = dict(
          a = 5,
          b = 20
    )
    
    def __init__(self):
        self.order = None
        
    def next(self):
        print("a = %d b = %d" % (self.p.a, self.p.b))


# 测试args
def test(*args, **kwargs):
    print(len(kwargs))
    for key in kwargs:
        print(key, kwargs[key])
        for i in kwargs[key]:
            print(i)


if __name__ == "__main__":
    start = "2018-01-01"
    end = "2020-07-05"
    name = ["300etf"]
    code = ["510300"]
    filename = code[0]+".csv"
    # 已有数据文件，直接读取数据
    if os.path.exists("./" + filename):
        df = pd.read_csv(filename)
    else: # 没有数据文件，用tushare下载
        df = ts.get_k_data(code, autype = "qfq", start = self.__start,  end = self.__end)
        df.to_csv(filename)
    df.index = pd.to_datetime(df.date)
    df['openinterest']=0
    df=df[['open','high','low','close','volume','openinterest']]
    start_date = list(map(int, start.split("-")))
    end_date = list(map(int, end.split("-")))
    start_date = datetime.datetime(start_date[0], start_date[1], start_date[2])
    end_date = datetime.datetime(end_date[0], end_date[1], end_date[2])
    dataFeed = bt.feeds.PandasData(dataname = df, name = name[0], fromdate = start_date, todate = end_date)
    cerebro = bt.Cerebro(maxcpus = 1)
    # cerebro.optstrategy(TestStrategy, a = range(1, 5), b = range(10, 15))
    cerebro.addstrategy(TestStrategy, a = 2, b = 15)
    cerebro.adddata(dataFeed, name = "test")
    cerebro.addanalyzer(btay.Returns, _name = "RE")
    results = cerebro.run()
    print(results[0].analyzers.RE.get_analysis()["rnorm"])
    
#    for result in results:
#        ret = result[0].analyzers.RE.get_analysis()["rnorm"]
#        print((result[0].p.a, result[0].p.b, ret))
        
    
    test(a = range(1, 5), b = (1, 5))
#    print(len(results))
#    print(results[0][0].p.a, results[0][0].p.b, results[0][0].analyzers.RE.get_analysis()["rnorm"])
        