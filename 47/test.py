# coding:utf-8
# 测试用的策略，每个交易日买入100股

import backtrader as bt
import backtest
import pandas as pd
import random
import backtest
import math


class TestStrategy(bt.Strategy):
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
    
    def __init__(self):
        self.order = None
        
    def next(self):
        self.order = self.buy(size = 100, price = self.datas[0].close)
        
    def stop(self):
        self.order = self.close()
        
        
# 制造测试数据
def makeTestData():
    df = pd.read_csv("510300.csv")
    df.index = pd.to_datetime(df.date)
    df['openinterest']=0
    df=df[['open','high','low','close','volume','openinterest']]
    # print(df.describe())
    # print(df.head())
    # 生成测试数据和基准数据
    testData = df.copy(deep = True)
    benchData = df.copy(deep = True)
    r = 0.02/252.0
    r_eps = 0.001 #上涨的幅度比下跌的高的部分
    i = 0
    base = 1
    
    for date in testData.index:
        oldvalue = base
        if i > 0:
            oldvalue = testData.iloc[i]["close"]
        rn = random.randint(0, 1)
        value = 0.0
        if rn == 0: # 上涨
            value = oldvalue * (1+r+r_eps)
        elif rn == 1: # 下跌
            value = oldvalue * (1-r)
        testData.loc[date.date(), "open"] = value
        testData.loc[date.date(), "high"] = value
        testData.loc[date.date(), "low"] = value
        testData.loc[date.date(), "close"] = value
        i += 1
    i = 0
    for date in benchData.index:
        value = base*r*math.pow(1+r, i)
        benchData.loc[date.date(), "open"] = value
        benchData.loc[date.date(), "high"] = value
        benchData.loc[date.date(), "low"] = value
        benchData.loc[date.date(), "close"] = value
        i += 1
    return testData, benchData
    
    
# 计算每日收益率
def getReturns(testData, benchData):
    testReturns = testData.pct_change()
    benchReturns = benchData.pct_change()
    return (testReturns.close[2:], benchReturns.close[2:])
        
        
if __name__ == "__main__":
#    start = "2018-01-01"
#    end = "2020-06-30"
#    name = ["pingan"]
#    code = ["000001"]
#    backtest = backtest.BackTest(TestStrategy, start, end, code, name, 1000000)
#    result = backtest.run()
#    # backtest.output()
#    print(result)
    # 生成数据
    test, bench = makeTestData()
    test.to_csv("test.csv")
    bench.to_csv("bench.csv")
    # 读取数据
    test = pd.read_csv("test.csv", index_col = "date")
    bench = pd.read_csv("bench.csv", index_col = "date")
    # print(test.describe(), bench.describe())
    test_returns, bench_returns =  getReturns(test, bench)
    # 计算回测指标
    risk = backtest.riskAnalyzer(test_returns, bench_returns, riskFreeRate = 0.02)
    results = risk.run()
    print("empyrical计算结果")
    print(results)
                