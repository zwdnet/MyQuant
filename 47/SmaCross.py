# coding:utf-8
# 双均线策略实现


import backtrader as bt
import backtest
import math


# 双均线策略类
class SmaCross(bt.Strategy):
    params = dict(
            pfast = 5,
            pslow = 30
    )
    
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
    
    def __init__(self):
        sma1 = bt.ind.SMA(period = self.p.pfast)
        sma2 = bt.ind.SMA(period = self.p.pslow)
        self.crossover = bt.ind.CrossOver(sma1, sma2)
        self.dataclose = self.datas[0].close
        self.order = None
        
    def next(self):
        if not self.position:
            if self.crossover > 0:
                cash = self.broker.get_cash()
                stock = math.ceil(cash/self.dataclose/100)*100 - 100
                self.order = self.buy(size = stock, price = self.datas[0].close)
                # self.log("买入")
        elif self.crossover < 0:
            self.order = self.close()
            # self.log("卖出")


if __name__ == "__main__":
    start = "2018-01-01"
    end = "2020-07-05"
    name = ["300etf"]
    code = ["510300"]
    backtest = backtest.BackTest(SmaCross, start, end, code, name, 10000)
    result = backtest.run()
    # backtest.output()
    print(result)
    