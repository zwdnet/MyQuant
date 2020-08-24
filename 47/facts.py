# coding:utf-8
# 多因子选股实现


import backtrader as bt
import backtrader.indicators as bi
import backtest
import pandas as pd
import tushare as ts
import matplotlib.pyplot as plt
from xpinyin import Pinyin
import datetime



# 获取股票数据，进行初步筛选，返回供因子分析的股票数据。
def getFactors():
#    data = ts.get_stock_basics()
#    print(data.head())
#    print(len(data))
#    data.to_csv("stocks.csv")
    data = pd.read_csv("stocks.csv", index_col = "code")
    # 排除亏损的股票
    data = data[data.npr > 0.0]
    # 排除上市不满2年的
    data = data[data.timeToMarket <= 20180801]
    # 排除ST股票
    data = data[~ data.name.str.contains("ST")]
    # 排除代码小于100000的股票
    data = data[data.index >= 100000]
    # print(data)
    return data


# 分析数据
def analysis(factors):
    print("平均市盈率:%.2f" % (factors.pe.mean()))
    print("每股收益:%.2f" % (factors.esp.mean()))
    print("每股净资产:%.2f" % (factors.bvps.mean()))
    print("平均市净率:%.2f" % (factors.pb.mean()))
    print("平均每股净利润:%.2f" % (factors.npr.mean()))
    print("平均股东人数:%.2f" % (factors.holders.mean()))
    # 绘图
    print(factors.pe)
    plt.figure()
    factors.pe.hist(bins = 100, range = (0, 2.0), align = "left")
    plt.savefig("PE.png")
    plt.figure()
    factors.esp.hist(bins = 100, range = (0, 2.0), align = "left")
    plt.savefig("ESP.png")
    plt.figure()
    factors.pb.hist(bins = 100, range = (0, 50.0), align = "left")
    plt.savefig("PB.png")
    plt.figure()
    factors.npr.hist(bins = 100, range = (0, 50.0), align = "left")
    plt.savefig("NPR.png")
    plt.figure()
    factors.holders.hist(bins = 100, range = (0, 50.0), align = "left")
    plt.savefig("HOLDERS.png")
    
    
# 计算评分指标
def scale(factors):
    pe = -1.0*factors.pe/factors.pe.mean()
    esp = factors.esp/factors.esp.mean()
    bvps = factors.bvps/factors.bvps.mean()
    pb = factors.pb/factors.pb.mean()
    npr = factors.npr/factors.npr.mean()
    score = pe+esp+bvps+pb+npr
    print(score)
    # 排序并画图
    score = score.sort_values()
    print(score)
    score.plot(kind = "hist", bins = 1000, range = (-25.0, 30.0))
    plt.savefig("fsctorScore.png")
    return score
    
    
# 交易策略类，一开始买入然后持有。
class FactorStrategy(bt.Strategy):
    def __init__(self):
        self.p_value = self.broker.getvalue()*0.9/10.0
        
    def next(self):
        # 买入
        for data in self.datas:
            # 获取仓位
            pos = self.getposition(data).size
            if pos == 0:
                size = int(self.p_value/100/data.close[0])*100
                self.buy(data = data, size = size)
        # 最后卖出
        date = self.datas[0].datetime.date(0)
        closeDate = datetime.datetime(2020, 7, 2)
        if date.year == closeDate.year and date.month == closeDate.month and date.day == closeDate.day:
            for data in self.datas:
                pos = self.getposition(data).size
                if pos != 0:
                    self.sell(data = data, size = pos )
                
    # 输出
    def log(self, txt):
        print(txt)
        
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
                
    # 交易情况
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.__displayOrder(True, order)
            elif order.issell():
                self.__displayOrder(False, order)
        self.order = None


if __name__ == "__main__":
    factors = getFactors()
    analysis(factors)
    score = scale(factors)
    codes = score[-10:].index
    # 进行回测
    start = "2018-01-01"
    end = "2020-07-05"
    name = factors.loc[codes, "name"].values
    # 将汉字转换为拼音
    p = Pinyin()
    name = [p.get_pinyin(s) for s in name]
    code = [str(x) for x in codes]
    print(len(name), code)
    backtest = backtest.BackTest(FactorStrategy, start, end, code, name, 1000000, bDraw = True)
    result = backtest.run()
    backtest.output()
    print(result)