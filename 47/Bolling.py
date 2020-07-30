# coding:utf-8
# 布林带策略实现


import backtrader as bt
import backtest
import pandas as pd
import math


class Bolling(bt.Strategy):
    params = dict(
            period = 50
    )
    
    def __init__(self):
        self.bb = bt.ind.BBands(period = self.p.period, devfactor = 2.0)
        self.dataclose = self.datas[0].close
        self.order = None
        
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
        
    def next(self):
        # self.log("持仓%d" % (self.position.size))
        if self.order:
            return
        if not self.position:
            # 突破下轨 买入
            if self.bb.bot > self.datas[0].close:
                cash = self.broker.get_cash()
                stock = math.ceil(cash/self.dataclose/100)*100 - 100
                self.order = self.buy(size = stock, price = self.datas[0].close, exectype = bt.Order.Market)
        else:
            # 持仓且突破上轨，卖出
            if self.bb.bot < self.datas[0].close:
                self.order = self.close()
            
    def notify_order(self, order):
        # 有交易提交/被接受，啥也不做
        if order.status in [order.Submitted, order.Accepted]:
            return

        self.order = None


if __name__ == "__main__":
    start = "2018-01-01"
    end = "2020-07-05"
    name = ["nasetf"]
    code = ["513100"]
    backtest = backtest.BackTest(Bolling, start, end, code, name, 10000)
    result = backtest.run()
    # backtest.output()
    print(result)
    result = backtest.optRun(period = range(6, 30))
    print(result)
    