# coding:utf-8
# 海龟策略实现


import backtrader as bt
import backtrader.indicators as bi
import backtest
import pandas as pd
import math
import matplotlib.pyplot as plt


class TurtleStrategy(bt.Strategy):
    params = (
        ("long_period", 20),
        ("short_period", 10),
        ("printlog", False),
    )
    
    def __init__(self):
        self.order = None
        self.buyprice = 0
        self.comm = 0
        self.buy_size = 0
        self.buy_count = 0
        # 用到的指标
        self.H_line = bi.Highest(self.data.high(-1), period = self.p.long_period)
        self.L_line = bi.Lowest(self.data.low(-1), period = self.p.long_period)
        self.TR = bi.Max((self.data.high(0) - self.data.low(0)), abs(self.data.close(-1) - self.data.high(0)), abs(self.data.close(-1) - self.data.low(0)))
        self.ATR = bi.SimpleMovingAverage(self.TR, period = 14)
        # 价格与上下轨线交叉
        self.buy_signal = bt.ind.CrossOver(self.data.close(0), self.H_line)
        self.sell_signal = bt.ind.CrossOver(self.data.close(0), self.L_line)
        
    def next(self):
        if self.order:
            return
            
        # 入场:价格突破上轨线且空仓时
        if self.buy_signal > 0 and self.buy_count == 0:
            self.buy_size = math.ceil((self.broker.getvalue() * 0.01 / self.ATR) / 100) * 100
            self.sizer.p.stake = self.buy_size
            self.buy_count = 1
            self.order = self.buy()
            self.log("入场")
            
        # 加仓: 价格上涨了买入价的0.5ATR且加仓次数少于3次(含)
        elif self.data.close > self.buyprice + 0.5*self.ATR[0] and self.buy_count > 0 and self.buy_count <= 4:
            self.buy_size = math.ceil((self.broker.get_cash() * 0.01 / self.ATR) / 100) * 100
            self.sizer.p.stake = self.buy_size
            self.order = self.buy()
            self.buy_count += 1
            self.log("加仓")
            
        # 离场: 价格跌破下轨线且持仓时
        elif self.sell_signal < 0 and self.buy_count > 0:
            self.order = self.sell()
            self.buy_count = 0
            self.log("离场")
            
        # 止损: 价格跌破买入价的2个ATR且持仓时
        elif self.data.close < (self.buyprice - 2*self.ATR[0]) and self.buy_count > 0:
            self.order = self.sell()
            self.buy_count = 0
            self.log("止损")
            
            
    # 输出交易记录
    def log(self, txt, dt = None, doprint = False):
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))
            
    def notify_order(self, order):
        # 有交易提交/被接受，啥也不做
        if order.status in [order.Submitted, order.Accepted]:
            return
        # 交易完成，报告结果
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    '执行买入, 价格: %.2f, 成本: %.2f, 手续费 %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))
                self.buyprice = order.executed.price
                self.comm += order.executed.comm
            else:
                self.log(
                    '执行卖出, 价格: %.2f, 成本: %.2f, 手续费 %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))
                self.comm += order.executed.comm
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("交易失败")
        self.order = None
        
    # 输出手续费
    def stop(self):
        self.log("手续费:%.2f 成本比例:%.5f" % (self.comm, self.comm/self.broker.getvalue()))
        
        
if __name__ == "__main__":
    start = "2018-01-01"
    end = "2020-07-05"
    name = ["300etf"]
    code = ["510300"]
    backtest = backtest.BackTest(TurtleStrategy, start, end, code, name, 100000)
    result = backtest.run()
    # backtest.output()
    print(result)
    result = backtest.optRun(long_period = range(20, 40), short_period = range(5, 15))
    plt.figure()
    plt.plot(result.参数值, result.年化收益率)
    plt.savefig("海龟策略参数优化.png")
    ret = result.loc[:, "年化收益率"]
    maxindex = ret[ret == ret.max()].index
    bestResult = result.loc[maxindex,:]
    print(bestResult.loc[:, ["夏普比率", "参数名", "参数值",  "年化收益率"]])
    