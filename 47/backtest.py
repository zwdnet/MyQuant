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
    def __init__(self, strategy, start, end, code, name, cash = 0.01, commission = 0.0003, benchmarkCode = "510300", bDraw = True):
        self.__cerebro = None
        self.__strategy = strategy
        self.__start = start
        self.__end = end
        self.__code = code
        self.__name = name
        self.__result = None
        self.__commission = commission
        self.__initcash = cash
        self.__backtestResult = pd.Series()
        self.__returns = pd.Series()
        self.__benchmarkCode = benchmarkCode
        self.__benchReturns = pd.Series()
        self.__benchFeed = None
        self.__bDraw = bDraw
        self.__start_date = None
        self.__end_date = None
        self.init()
        
    # 真正进行初始化的地方
    def init(self):
        self.__cerebro = bt.Cerebro()
        self.__cerebro.addstrategy(self.__strategy)
        self.createDataFeeds()
        self.settingCerebro()
        
    # 进行参数优化
    def _optStrategy(self, *args, **kwargs):
        self.__cerebro = bt.Cerebro(maxcpus = 1)
        self.__cerebro.optstrategy(self.__strategy, *args, **kwargs)
        self.createDataFeeds()
        self.settingCerebro()

        
    # 设置cerebro
    def settingCerebro(self):
        # 添加回撤观察器
        self.__cerebro.addobserver(bt.observers.DrawDown)
        # 添加基准观察器
        self.__cerebro.addobserver(bt.observers.Benchmark, data = self.__benchFeed, timeframe = bt.TimeFrame.NoTimeFrame)
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
        self.__cerebro.addanalyzer(btay.TimeReturn, _name = "TR_Bench", data = self.__benchFeed)
        self.__cerebro.addanalyzer(btay.SQN, _name = "SQN")
        
    # 建立数据源
    def createDataFeeds(self):
        # 建立回测数据源
        for i in range(len(self.__code)):
            dataFeed = self._createDataFeeds(self.__code[i], self.__name[i])
            self.__cerebro.adddata(dataFeed, name = self.__name[i])
        self.__benchFeed = self._createDataFeeds(self.__benchmarkCode, "benchMark")
        self.__cerebro.adddata(self.__benchFeed, name = "benchMark")
            
    # 建立数据源的具体过程
    def _createDataFeeds(self, code, name):
        df_data = self._getData(code)
        start_date = list(map(int, self.__start.split("-")))
        end_date = list(map(int, self.__end.split("-")))
        self.__start_date = datetime.datetime(start_date[0], start_date[1], start_date[2])
        self.__end_date = datetime.datetime(end_date[0], end_date[1], end_date[2])
        dataFeed = bt.feeds.PandasData(dataname = df_data, name = name, fromdate = datetime.datetime(start_date[0], start_date[1], start_date[2]), todate = datetime.datetime(end_date[0], end_date[1], end_date[2]))
        return dataFeed
            
    # 获取账户总价值
    def getValue(self):
        return self.__cerebro.broker.getvalue()
        
    # 计算胜率信息
    def _winInfo(self, trade_info, result):
        total_trade_num = trade_info["total"]["total"]
        # print(total_trade_num)
        if total_trade_num > 1:
            win_num = trade_info["won"]["total"]
            lost_num = trade_info["lost"]["total"]
            result["交易次数"] = total_trade_num
            result["胜率"] = win_num/total_trade_num
            result["败率"] = lost_num/total_trade_num
            
    # 根据SQN值对策略做出评估
    # 按照backtrader文档写的
    def _judgeBySQN(self, sqn):
        result = None
        if sqn >= 1.6 and sqn <= 1.9:
            result = "低于平均"
        elif sqn > 1.9 and sqn <= 2.4:
            result = "平均水平"
        elif sqn > 2.4 and sqn <= 2.9:
            result = "良好"
        elif sqn > 2.9 and sqn <= 5.0:
            result = "优秀"
        elif sqn > 5.0 and sqn <= 6.9:
            result = "卓越"
        elif sqn > 6.9:
            result = "大神?"
        else:
            result = "很差"
        self.__backtestResult["策略评价(根据SQN)"] = result
        return result
        
    # 计算并保存回测结果指标
    def _Result(self):
        self.__backtestResult["账户总额"] = self.getValue()
        self.__backtestResult["总收益率"] = self.__results[0].analyzers.RE.get_analysis()["rtot"]
        self.__backtestResult["年化收益率"] = self.__results[0].analyzers.RE.get_analysis()["rnorm"]
        # self.__backtestResult["交易成本"] = self.__cerebro.strats[0].getCommission()
        self.__backtestResult["夏普比率"] = self.__results[0].analyzers.sharpe.get_analysis()["sharperatio"]
        self.__backtestResult["最大回撤"] = self.__results[0].analyzers.DD.get_analysis().max.drawdown
        self.__backtestResult["最大回撤期间"] = self.__results[0].analyzers.DD.get_analysis().max.len
        self.__backtestResult["SQN"] = self.__results[0].analyzers.SQN.get_analysis()["sqn"]
        self._judgeBySQN(self.__backtestResult["SQN"])

        # 计算胜率信息
        trade_info = self.__results[0].analyzers.TA.get_analysis()
        self._winInfo(trade_info, self.__backtestResult)
        
    # 取得优化参数时的指标结果
    def _getOptAnalysis(self, result):
        temp = dict()
        temp["总收益率"] = result[0].analyzers.RE.get_analysis()["rtot"]
        temp["年化收益率"] = result[0].analyzers.RE.get_analysis()["rnorm"]
        temp["夏普比率"] = result[0].analyzers.sharpe.get_analysis()["sharperatio"]
        temp["最大回撤"] = result[0].analyzers.DD.get_analysis().max.drawdown
        temp["最大回撤期间"] = result[0].analyzers.DD.get_analysis().max.len
        sqn = result[0].analyzers.SQN.get_analysis()["sqn"]
        temp["SQN"] = sqn
        temp["策略评价(根据SQN)"] = self._judgeBySQN(sqn)
        trade_info = self.__results[0].analyzers.TA.get_analysis()
        self._winInfo(trade_info, temp)
        return temp
        
    # 在优化参数时计算并保存回测结果
    def _optResult(self, results, **kwargs):
        testResults = pd.DataFrame()
        params = []
        for k, v in kwargs.items():
            for t in v:
                 params.append(t)
        i = 0
        for result in results:
            temp = self._getOptAnalysis(result)
            temp["参数值"] = params[i]
            i += 1
            returns = self._timeReturns(result)
            benchReturns = self._getBenchmarkReturns(result)
            self._riskAnaly(returns, benchReturns, temp)
            testResults = testResults.append(temp, ignore_index=True)
        testResults.set_index(["参数值"], inplace = True)
        return testResults
        
    # 获取回测指标
    def getResult(self):
        return self.__backtestResult
        
    # 获取策略及基准策略收益率的序列
    def getReturns(self):
        return self.__returns, self.__benchReturns
        
    # 计算收益率序列
    def _timeReturns(self, result):
        return pd.Series(result[0].analyzers.TR.get_analysis())
        
    # 运行基准策略，获取基准收益值
    def _getBenchmarkReturns(self, result):
        return pd.Series(result[0].analyzers.TR_Bench.get_analysis())
        
    # 分析策略的风险指标
    def _riskAnaly(self, returns, benchReturns, results):
        risk = riskAnalyzer(returns, benchReturns)
        result = risk.run()
        results["阿尔法"] = result["阿尔法"]
        results["贝塔"] = result["贝塔"]
        results["信息比例"] = result["信息比例"]
        results["策略波动率"] = result["策略波动率"]
        results["欧米伽"] = result["欧米伽"]
        # self.__backtestResult["夏普值"] = result["夏普值"]
        results["sortino"] = result["sortino"]
        results["calmar"] = result["calmar"]
        
    # 执行回测
    def run(self):
        self.__backtestResult["期初账户总值"] = self.getValue()
        self.__results = self.__cerebro.run()
        self.__backtestResult["期末账户总值"] = self.getValue()
        self._Result()
        if self.__bDraw == True:
            self._drawResult()
        self.__returns = self._timeReturns(self.__results)
        self.__benchReturns = self._getBenchmarkReturns(self.__results)
        self._riskAnaly(self.__returns, self.__benchReturns, self.__backtestResult)
        return self.getResult()
        
    # 执行参数优化的回测
    def optRun(self, *args, **kwargs):
        self._optStrategy(*args, **kwargs)
        results = self.__cerebro.run()
        testResults = self._optResult(results, **kwargs)
        self.init()
        return testResults
        
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
    