# coding:utf-8
# 网格交易策略实现


import backtrader as bt
import backtrader.indicators as bi
import backtest
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


class GridStrategy(bt.Strategy):
    params = (
        ("printlog", True),
        ("top", 4.2),
        ("buttom", 3.5),
        ("gap", 0.01)
    )
    def __init__(self):
        self.mid = (self.p.top + self.p.buttom)/2.0
        # 百分比区间计算
        #这里多1/2，是因为arange函数是左闭右开区间。
        perc_level = [x for x in np.arange(1 + self.p.gap * 5, 1 - self.p.gap * 5 - self.p.gap/2, -1.0 * self.p.gap)]
        # 价格区间
        # print(self.mid)
        self.price_levels = [self.mid * x for x in perc_level]
        # 记录上一次穿越的网格
        self.last_price_index = None
        # 总手续费
        self.comm = 0.0
        
    def next(self):
        # print(self.last_price_index)
        # 开仓
        if self.last_price_index == None:
            # print("b", len(self.price_levels))
            for i in range(len(self.price_levels)):
                price = self.data.close[0]
                # print("c", i, price, self.price_levels[i][0])
                if self.data.close[0] > self.price_levels[i]:
                    self.last_price_index = i
                    self.order_target_percent(target=i/(len(self.price_levels) - 1))
                    print("a")
                    return
        # 调仓
        else:
            signal = False
            while True:
                upper = None
                lower = None
                if self.last_price_index > 0:
                    upper = self.price_levels[self.last_price_index - 1]
                if self.last_price_index < len(self.price_levels) - 1:
                    lower = self.price_levels[self.last_price_index + 1]
                # 还不是最轻仓，继续涨，再卖一档
                if upper != None and self.data.close > upper:
                    self.last_price_index = self.last_price_index - 1
                    signal = True
                    continue
                # 还不是最重仓，继续跌，再买一档
                if lower != None and self.data.close < lower:
                    self.last_price_index = self.last_price_index + 1
                    signal = True
                    continue
                break
            if signal:
                self.long_short = None
                self.order_target_percent(target=self.last_price_index/(len(self.price_levels) - 1))
                
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
    backtest = backtest.BackTest(GridStrategy, start, end, code, name, 100000)
    result = backtest.run()
    # backtest.output()
    print(result)
    # 看选择不同网格宽度的效果
    result = backtest.optRun(gap = np.arange(0.005, 0.055, 0.005))
    plt.figure()
    plt.plot(result.参数值, result.年化收益率)
    plt.savefig("网格策略宽度优化.png")
    ret = result.loc[:, "年化收益率"]
    maxindex = ret[ret == ret.max()].index
    bestResult = result.loc[maxindex,:]
    print(bestResult.loc[:, ["夏普比率", "参数名", "参数值",  "年化收益率"]])