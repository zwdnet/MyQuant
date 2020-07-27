# coding:utf-8
# 量化交易回测类


import backtrader as bt
import backtrader.analyzers as btay
import tushare as ts
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import empyrical as ey
import math
import tushare as ts
import numpy as np
from scipy import stats
from backtrader.utils.py3 import map
from backtrader import Analyzer, TimeFrame
from backtrader.mathsupport import average, standarddev
from backtrader.analyzers import AnnualReturn
import operator


# 回测类
class BackTest:
    def __init__(self, strategy, start, end, code, name, cash = 0.01, benchmarkCode = "510300", bDraw = True):
        self.__cerebro = None
        self.__strategy = strategy
        self.__start = start
        self.__end = end
        self.__code = code
        self.__name = name
        self.__result = None
        self.__commission = 0.0003
        self.__initcash = cash
        self.__backtestResult = pd.Series()
        self.__returns = pd.Series()
        self.__benchmarkCode = benchmarkCode
        self.__benchReturns = pd.Series()
        self.__bDraw = bDraw
        self.__start_date = None
        self.__end_date = None
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
        # 设置初始资金
        self.__cerebro.broker.setcash(self.__initcash)
        # 添加分析对象
        self.__cerebro.addanalyzer(btay.SharpeRatio, _name = "sharpe", riskfreerate = 0.02, stddev_sample = True, annualize = True)
        self.__cerebro.addanalyzer(btay.AnnualReturn, _name = "AR")
        self.__cerebro.addanalyzer(btay.DrawDown, _name = "DD")
        self.__cerebro.addanalyzer(btay.Returns, _name = "RE")
        self.__cerebro.addanalyzer(btay.TradeAnalyzer, _name = "TA")
        self.__cerebro.addanalyzer(btay.TimeReturn, _name = "TR")
        self.__cerebro.addanalyzer(btay.SQN, _name = "SQN")
        
    # 建立数据源
    def createDataFeeds(self):
        for i in range(len(self.__code)):
            df_data = self._getData(self.__code[i])
            start_date = list(map(int, self.__start.split("-")))
            end_date = list(map(int, self.__end.split("-")))
            self.__start_date = datetime.datetime(start_date[0], start_date[1], start_date[2])
            self.__end_date = datetime.datetime(end_date[0], end_date[1], end_date[2])
            dataFeed = bt.feeds.PandasData(dataname = df_data, name = self.__name[i], fromdate = datetime.datetime(start_date[0], start_date[1], start_date[2]), todate = datetime.datetime(end_date[0], end_date[1], end_date[2]))
            self.__cerebro.adddata(dataFeed, name = self.__name[i])
            
    # 获取账户总价值
    def getValue(self):
        return self.__cerebro.broker.getvalue()
        
    # 计算胜率信息
    def _winInfo(self):
        trade_info = self.__results[0].analyzers.TA.get_analysis()
        total_trade_num = trade_info["total"]["total"]
        # print(total_trade_num)
        if total_trade_num > 1:
            win_num = trade_info["won"]["total"]
            lost_num = trade_info["lost"]["total"]
            self.__backtestResult["交易次数"] = total_trade_num
            self.__backtestResult["胜率"] = win_num/total_trade_num
            self.__backtestResult["败率"] = lost_num/total_trade_num
        
        
    # 计算并保存回测结果指标
    def _Result(self):
        self.__backtestResult["账户总额"] = self.getValue()
        self.__backtestResult["总收益率"] = self.__results[0].analyzers.RE.get_analysis()["rtot"]
        self.__backtestResult["年化收益率"] = self.__results[0].analyzers.RE.get_analysis()["rnorm"]
        self.__backtestResult["夏普比率"] = self.__results[0].analyzers.sharpe.get_analysis()["sharperatio"]
        self.__backtestResult["最大回撤"] = self.__results[0].analyzers.DD.get_analysis().max.drawdown
        self.__backtestResult["最大回撤期间"] = self.__results[0].analyzers.DD.get_analysis().max.len
        self.__backtestResult["SQN"] = self.__results[0].analyzers.SQN.get_analysis()["sqn"]

        # 计算胜率信息
        self._winInfo()
        
    # 获取回测指标
    def getResult(self):
        return self.__backtestResult
        
    # 获取策略及基准策略收益率的序列
    def getReturns(self):
        return self.__returns, self.__benchReturns
        
    # 获取收益率序列
    def _timeReturns(self):
        self.__returns = pd.Series(self.__results[0].analyzers.TR.get_analysis())
        
    # 分析策略的风险指标
    def _riskAnaly(self):
        risk = riskAnalyzer(self.__returns, self.__benchReturns)
        result = risk.run()
        self.__backtestResult["阿尔法"] = result["阿尔法"]
        self.__backtestResult["贝塔"] = result["贝塔"]
        self.__backtestResult["信息比例"] = result["信息比例"]
        self.__backtestResult["策略波动率"] = result["策略波动率"]
        self.__backtestResult["欧米伽"] = result["欧米伽"]
        # self.__backtestResult["夏普值"] = result["夏普值"]
        self.__backtestResult["sortino"] = result["sortino"]
        self.__backtestResult["calmar"] = result["calmar"]
        
    # 执行回测
    def run(self):
        print("初始资金:%.2f" % self.getValue())
        self.__results = self.__cerebro.run()
        print("期末资金:%.2f" % self.getValue())
        self._Result()
        if self.__bDraw == True:
            self._drawResult()
        self._timeReturns()
        self.__benchReturns = self._runBenchmark()
        self._riskAnaly()
        return self.getResult()
        
    # 回测结果绘图
    def _drawResult(self):
        self.__cerebro.plot(numfigs = 2)
        figname = type(self).__name__+".png"
        plt.savefig(figname)
        
    # 输出回测结果
    def output(self):
        print("夏普比例:", self.__results[0].analyzers.sharpe.get_analysis()["sharperatio"])
        print("年化收益率:", self.__results[0].analyzers.AR.get_analysis())
        print("最大回撤:%.2f，最大回撤周期%d" % (self.__results[0].analyzers.DD.get_analysis().max.drawdown, self.__results[0].analyzers.DD.get_analysis().max.len))
        print("总收益率:%.2f" % (self.__results[0].analyzers.RE.get_analysis()["rtot"]))
        self.__results[0].analyzers.TA.pprint()
            
    # 获取数据
    def _getData(self, code):
        filename = code+".csv"
        # 已有数据文件，直接读取数据
        if os.path.exists("./" + filename):
            df = pd.read_csv(filename)
        else: # 没有数据文件，用tushare下载
            df = ts.get_k_data(code, autype = "qfq", start = self.__start,  end = self.__end)
            df.to_csv(filename)
        df.index = pd.to_datetime(df.date)
        df['openinterest']=0
        df=df[['open','high','low','close','volume','openinterest']]
        return df
        
    # 运行基准策略，获取基准收益值
    def _runBenchmark(self):
        benchCerebro = bt.Cerebro()
        benchCerebro.addstrategy(Benchmark)
        # 设置手续费
        self.__cerebro.broker.setcommission(commission=self.__commission)
        # 设置初始资金
        self.__cerebro.broker.setcash(self.__initcash)
        # 添加收益分析器
        benchCerebro.addanalyzer(btay.TimeReturn, _name = "TR")
        # 获取数据并添加
        df_data = self._getData(self.__benchmarkCode)
        start_date = list(map(int, self.__start.split("-")))
        end_date = list(map(int, self.__end.split("-")))
        dataFeed = bt.feeds.PandasData(dataname = df_data, name = "BenchMark", fromdate = datetime.datetime(start_date[0], start_date[1], start_date[2]), todate = datetime.datetime(end_date[0], end_date[1], end_date[2]))
        benchCerebro.adddata(dataFeed, name = "BenchMark")
        # 执行回测
        results = benchCerebro.run()
        # 获取收益率序列
        return pd.Series(results[0].analyzers.TR.get_analysis())
        
        
# 基准策略类，用于计算α，β等回测指标
# 采用第一天全仓买入并持有的策略
class Benchmark(bt.Strategy):
    def __init__(self):
        self.order = None
        self.bBuy = False
        self.dataclose = self.datas[0].close
        
    def next(self):
        if self.bBuy == True:
            return
        else:
            cash = self.broker.get_cash()
            stock = math.ceil(cash/self.dataclose/100)*100 - 100
            self.order = self.buy(size = stock, price = self.datas[0].close)
            self.bBuy = True
            
    def stop(self):
        self.order = self.close()
        
        
# 计算年化夏普值，参考backtrader的文档
class SharpeRatio(Analyzer):
    params = (("timeframe", TimeFrame.Years), ("riskfreerate", 0.02))
    
    def __init__(self):
        super(SharpeRatio, self).__init__()
        self.anret = AnnualReturn()
        
    def start(self):
        pass
        
    def next(self):
        pass
        
    def stop(self):
        retfree = [self.p.riskfreerate] * len(self.anret.rets)
        retavg = average(list(map(operator.sub, self.anret.rets, retfree)))
        retdev = standarddev(self.anret.rets)
        if retdev == 0.0:
            self.ratio = 0.0
        else:
            self.ratio = retavg/retdev
        
    def get_analysis(self):
        return dict(sharperatio = self.ratio)
        

# 用empyrical库计算风险指标
class riskAnalyzer:
    def __init__(self, returns, benchReturns, riskFreeRate = 0.02):
        self.__returns = returns
        self.__benchReturns = benchReturns
        self.__risk_free = riskFreeRate
        self.__alpha = 0.0
        self.__beta = 0.0
        self.__info = 0.0
        self.__vola = 0.0
        self.__omega = 0.0
        self.__sharpe = 0.0
        self.__sortino = 0.0
        self.__calmar = 0.0
        
    def run(self):
        # 计算各指标
        self._alpha_beta()
        self._info()
        self._vola()
        self._omega()
        self._sharpe()
        self._sortino()
        self._calmar()
        result = pd.Series(dtype = "float64")
        result["阿尔法"] = self.__alpha
        result["贝塔"] = self.__beta
        result["信息比例"] = self.__info
        result["策略波动率"] = self.__vola
        result["欧米伽"] = self.__omega
        result["夏普值"] = self.__sharpe
        result["sortino"] = self.__sortino
        result["calmar"] = self.__calmar
        return result
        
    def _alpha_beta(self):
        self.__alpha, self.__beta = ey.alpha_beta(returns = self.__returns, factor_returns = self.__benchReturns, risk_free = self.__risk_free, annualization = 1)
        
    def _info(self):
        self.__info = ey.excess_sharpe(returns = self.__returns, factor_returns = self.__benchReturns)
        
    def _vola(self):
        self.__vola = ey.annual_volatility(self.__returns, period='daily')
    
    def _omega(self):
        self.__omega = ey.omega_ratio(returns = self.__returns, risk_free = self.__risk_free)
        
    def _sharpe(self):
        self.__sharpe = ey.sharpe_ratio(returns = self.__returns, annualization = 1)
        
    def _sortino(self):
        self.__sortino = ey.sortino_ratio(returns = self.__returns)
        
    def _calmar(self):
        self.__calmar = ey.calmar_ratio(returns = self.__returns)
        
        
# 测试函数
def test():
    # 构造测试数据
    returns = pd.Series(
        index = pd.date_range("2017-03-10", "2017-03-19"),
        data = (-0.012143, 0.045350, 0.030957, 0.004902, 0.002341, -0.02103, 0.00148, 0.004820, -0.00023, 0.01201))
    print(returns)
    benchmark_returns = pd.Series(
        index = pd.date_range("2017-03-10", "2017-03-19"),
        data = ( -0.031940, 0.025350, -0.020957, -0.000902, 0.007341, -0.01103, 0.00248, 0.008820, -0.00123, 0.01091))
    print(benchmark_returns)
    # 计算累积收益率
    creturns = ey.cum_returns(returns)
    print("累积收益率\n", creturns)
    risk = riskAnalyzer(returns, benchmark_returns, riskFreeRate = 0.01)
    results = risk.run()
    print(results)
    # 直接调用empyrical试试
    alpha = ey.alpha(returns = returns, factor_returns = benchmark_returns, risk_free = 0.01)
    calmar = ey.calmar_ratio(returns)
    print(alpha, calmar)
    # 自己计算阿尔法值
    annual_return = ey.annual_return(returns)
    annual_bench = ey.annual_return(benchmark_returns)
    print(annual_return, annual_bench)
    alpha2 = (annual_return - 0.01) - results["贝塔"]*(annual_bench - 0.01)
    print(alpha2)
    # 自己计算阿尔法贝塔
    def get_return(code, startdate, endate):
        df = ts.get_k_data(code, ktype = "D", autype = "qfq", start = startdate, end = endate)
        p1 = np.array(df.close[1:])
        p0 = np.array(df.close[:-1])
        logret = np.log(p1/p0)
        rate = pd.DataFrame()
        rate[code] = logret
        rate.index = df["date"][1:]
        return rate
    def alpha_beta(code, startdate, endate):
        mkt_ret = get_return("sh", startdate, endate)
        stock_ret = get_return(code, startdate, endate)
        df = pd.merge(mkt_ret, stock_ret, left_index = True, right_index = True)
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
        beta, alpha, r_value, p_value, std_err = stats.linregress(x, y)
        return (alpha, beta)
    def stocks_alpha_beta(stocks, startdate, endate):
        df = pd.DataFrame()
        alpha = []
        beta = []
        for code in stocks.values():
            a, b = alpha_beta(code, startdate, endate)
            alpha.append(float("%.4f"%a))
            beta.append(float("%.4f"%b))
        df["alpha"] = alpha
        df["beta"] = beta
        df.index = stocks.keys()
        return df
        
    startdate = "2017-01-01"
    endate = "2018-11-09"
    stocks={'中国平安':'601318','格力电器':'000651','招商银行':'600036','恒生电子':'600570','中信证券':'600030','贵州茅台':'600519'}
    results = stocks_alpha_beta(stocks, startdate, endate)
    print("自己计算结果")
    print(results)
    
    # 用empyrical计算
    def stocks_alpha_beta2(stocks, startdate, endate):
        df = pd.DataFrame()
        alpha = []
        beta = []
        for code in stocks.values():
            a, b = empyrical_alpha_beta(code, startdate, endate)
            alpha.append(float("%.4f"%a))
            beta.append(float("%.4f"%b))
        df["alpha"] = alpha
        df["beta"] = beta
        df.index = stocks.keys()
        return df
    def empyrical_alpha_beta(code, startdate, endate):
        mkt_ret = get_return("sh", startdate, endate)
        stock_ret = get_return(code, startdate, endate)
        alpha, beta = ey.alpha_beta(returns = stock_ret, factor_returns = mkt_ret,  annualization = 1)
        return (alpha, beta)
        
    results2 = stocks_alpha_beta2(stocks, startdate, endate)
    print("empyrical计算结果")
    print(results2)
    print(results2["alpha"]/results["alpha"])
    
    

# 测试夏普值的计算
def testSharpe():
    # 读取数据
    stock_data = pd.read_csv("stock_data.csv", parse_dates = ["Date"], index_col = ["Date"]).dropna()
    benchmark_data = pd.read_csv("benchmark_data.csv", parse_dates = ["Date"], index_col = ["Date"]).dropna()
    # 了解数据
    print("Stocks\n")
    print(stock_data.info())
    print(stock_data.head())
    print("\nBenchmarks\n")
    print(benchmark_data.info())
    print(benchmark_data.head())
    # 输出统计量
    print(stock_data.describe())
    print(benchmark_data.describe())
    # 计算每日回报率
    stock_returns = stock_data.pct_change()
    print(stock_returns.describe())
    sp_returns = benchmark_data.pct_change()
    print(sp_returns.describe())
    # 每日超额回报
    excess_returns = pd.DataFrame()
    risk_free = 0.04/252.0
    excess_returns["Amazon"] = stock_returns["Amazon"] - risk_free
    excess_returns["Facebook"] = stock_returns["Facebook"] - risk_free
    print(excess_returns.describe())
    # 超额回报的均值
    avg_excess_return = excess_returns.mean()
    print(avg_excess_return)
    # 超额回报的标准差
    std_excess_return = excess_returns.std()
    print(std_excess_return)
    # 计算夏普比率
    # 日夏普比率
    daily_sharpe_ratio = avg_excess_return.div(std_excess_return)
    # 年化夏普比率
    annual_factor = np.sqrt(252)
    annual_sharpe_ratio = daily_sharpe_ratio.mul(annual_factor)
    print("年化夏普比率\n", annual_sharpe_ratio)
    
    # 用empyrical算
    sharpe = pd.DataFrame()
    a = ey.sharpe_ratio(stock_returns["Amazon"],  risk_free = risk_free)#, annualization = 252)
    b = ey.sharpe_ratio(stock_returns["Facebook"], risk_free = risk_free)
    print("empyrical计算结果")
    print(a, b)
    print(a/annual_sharpe_ratio["Amazon"], b/annual_sharpe_ratio["Facebook"])
    
if __name__ == "__main__":
    # test()
    testSharpe()
    