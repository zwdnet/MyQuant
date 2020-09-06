# coding:utf-8
# 多因子选股, 用机器学习算法


import backtrader as bt
import backtrader.indicators as bi
import backtest
import pandas as pd
import os
import tushare as ts
import matplotlib.pyplot as plt
from xpinyin import Pinyin
import datetime
import random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals import joblib
import numpy as np
# from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
# from DL import DL


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
    # data = data[data.index >= 100000]
    # 排除退市的股票
    data = data[data.pe != 0]
    return data
    
    
# 根据股票代码找股票名称
def fromCodeToName(factors, codes):
    # 准备数据
    name = factors[factors.index.isin(codes)].name.values
    # 将汉字转换为拼音
    p = Pinyin()
    names = [p.get_pinyin(s) for s in name]
    return names
        
    
# 对所有股票回测其年化收益率
def getReturn(data):
    if os.path.exists("data.csv"):
        data = pd.read_csv("data.csv", index_col = "code")
        return data
    start = "2017-01-01"
    end = "2020-07-31"
    codes = data.index
    names = fromCodeToName(data, codes)
    # codes = [str(x) for x in codes]
    codes = ["%06d" % x for x in codes]
    print(codes)
#    print(codes)
#    print(names)
    # 在数据中增加一列计算年化收益率
    data["ar"] = 0.0
    t = 0
    cash = 100000
    # 到这里出错，移除。
    codes.remove("000029")
    for code in codes:
        test = backtest.BackTest(FactorStrategy, start, end, [code], [names[t]], cash, bDraw = False)
        result = test.run()
        print("第{}次回测，股票代码{}，回测年化收益率{}。".format(t+1, code, result.年化收益率))
        data.loc[code, ["ar"]] = result.年化收益率
        t += 1
    data.to_csv("data.csv")
    return data


# 分析数据
def analysis(data):
    print(data.info())
    # 收益率分布情况
    plt.figure()
    data.ar.hist(bins = 100)
    plt.savefig("returns.png")
    plt.close()
    print(data.ar.max())
    # 画图看各数据关系。
    # g = sns.pairplot(data)
    # g.savefig("data.png")
    
    
# 测试所选股票的回测结果
def doTest(codes, method):
    names = fromCodeToName(data, codes)
    codes = [str(x) for x in codes]
    start = "2010-01-01"
    end = "2020-07-01"
    cash = 1000000
    opttest = backtest.BackTest(FactorStrategy, start, end, codes, names, cash, bDraw = False)
    result = opttest.run()
    result["选股方法"] = method
    result["股票代码"] = codes
    print(result)
    return result
    
    
# 对数据打标签
def addItem(data):
    data["res"] = 0
    data.loc[data.ar > 0.0, "res"] = 1
    data.loc[data.ar <= 0.0, "res"] = 0
    return data
    
    
# 多元线性回归
def multiLineRegress(data):
    x = data.iloc[:, 3:21]
    y = data.iloc[:, 22]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 631)
    line_reg = LinearRegression(normalize = True)
    model = line_reg.fit(x_train, y_train)
    print("模型参数:", model)
    print("模型截距:", model.intercept_)
    print("参数权重:", model.coef_)
    
    y_pred = line_reg.predict(x_test)
    sum_mean = 0
    for i in range(len(y_pred)):
        sum_mean += (y_pred[i] - y_test.values[i]) ** 2
    sum_erro = np.sqrt(sum_mean /len(y_pred))
    print("RMSR=", sum_erro)
    print("Score=", model.score(x_test, y_test))
    # ROC曲线
    plt.figure()
    plt.plot(range(len(y_pred)), y_pred, 'b', label="predict")
    plt.plot(range(len(y_pred)), y_test, 'r', label="test")
    plt.legend(loc="upper right") 
    # 显示图中的标签
    plt.xlabel("facts")
    plt.ylabel('ar')
    plt.savefig("line_regress_result.png")
    plt.close()
    # 保存模型
    joblib.dump(model, "LineRegress.m")
    return model
    
    
# 用二次多项式回归进行选股
def quadRegress(data):
    quadratic_featurizer = PolynomialFeatures(degree=2)
    x = data.iloc[:, 3:21]
    y = data.iloc[:, 22]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 631)
    x_train_quad = quadratic_featurizer.fit_transform(x_train)
    x_test_quad = quadratic_featurizer.transform(x_test)
    regress_quad = LinearRegression()
    regress_quad.fit(x_train_quad, y_train)
    pred = regress_quad.predict(x_test_quad)
    print("2次多项式回归结果:", regress_quad.score(x_test_quad, y_test))
    plt.figure()
    plt.plot(range(len(pred)), y_test, "b", label="predict")
    plt.plot(range(len(pred)), pred, "r", label="test")
    plt.savefig("二次多项式回归.png")
    plt.close()
    # 保存模型
    joblib.dump(regress_quad, "QuadRegress.m")
    return regress_quad
    
    
# 测试二项式回归结果
def testQuadRegress(data, modelfile, modelname):
    quadratic_featurizer = PolynomialFeatures(degree=2)
    model = joblib.load(modelfile)
    pred_return = model.predict(quadratic_featurizer.fit_transform(data.iloc[:, 3:21]))
    # print(pred_return)
    data["pred_ar"] = pred_return
    # print(data)
    # 排序
    data = data.sort_values(by = "pred_ar", ascending = False)
    # print(data)
    # 取前十个股票作为投资标的
    codes = data.index[0:10].values
    # print(codes)
    return doTest(codes, method = modelname)
    

# 随机森林进行选股
def RFRegress(data):
    x = data.iloc[:, 3:21]
    y = data.iloc[:, 22]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 631)
    rfr = RandomForestRegressor()
    rfr.fit(x_train, y_train)
    pred = rfr.predict(x_test)
    # print(pred)
    plt.figure()
    plt.plot(range(len(pred)), y_test, "b", label="predict")
    plt.plot(range(len(pred)), pred, "r", label="test")
    plt.savefig("随机森林回归.png")
    plt.close()
    # 保存模型
    joblib.dump(rfr, "RandomForestRegress.m")
    return rfr
    
    
# 感知机模型分析
def doPerceptron(data):
    x = data.iloc[:, 3:21]
    y = data.loc[:, ["res"]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 631)
    clf = Perceptron(fit_intercept = False, shuffle = False)
    #clf = Perceptron(fit_intercept = False, n_iter = 30, shuffle = False)
    clf.fit(x_train, y_train.astype("int"))
    print("感知机得分:", clf.score(x_test, y_test))
    pred = clf.predict(x_test)
    # print(pred)
    plt.figure()
    plt.scatter(range(len(pred)), y_test, c = "b", label="predict")
    plt.scatter(range(len(pred)), pred, c = "r", label="test")
    plt.savefig("感知机模型分类.png")
    plt.close()
    # 保存模型
    joblib.dump(clf, "Perceptron.m")
    return clf
    
    
# 支持向量机选股
def SVCClassic(data):
    x = data.iloc[:, 3:21]
    y = data.loc[:, ["res"]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 631)
    clf = SVC()
    clf.fit(x_train, y_train)
    print("支持向量机评分:", clf.score(x_test, y_test))
    pred = clf.predict(x_test)
    # print(pred)
    plt.figure()
    plt.scatter(range(len(pred)), y_test, c = "b", label="predict")
    plt.scatter(range(len(pred)), pred, c = "r", label="test")
    plt.savefig("支持向量机分类.png")
    plt.close()
    # 保存模型
    joblib.dump(clf, "SVC.m")
    return clf
    
    
# 逻辑回归进行多因子选股
def logist(data):
    x = data.iloc[:, 3:21]
    y = data.loc[:, ["res"]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 631)
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    print("逻辑回归评分:", clf.score(x_test, y_test))
    pred = clf.predict(x_test)
    # print(pred)
    plt.figure()
    plt.scatter(range(len(pred)), y_test, c = "b", label="predict")
    plt.scatter(range(len(pred)), pred, c = "r", label="test")
    plt.savefig("逻辑回归分类.png")
    plt.close()
    # 保存模型
    joblib.dump(clf, "Logistic.m")
    return clf
    
    
# KNN聚类
def KNN(data):
    x = data.iloc[:, 3:21]
    y = data.loc[:, ["res"]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 631)
    knn = KNeighborsClassifier() 
    knn.fit(x_train, y_train)
    print("KNN评分:", knn.score(x_test, y_test))
    pred = knn.predict(x_test)
    # print(pred)
    plt.figure()
    plt.scatter(range(len(pred)), y_test, c = "b", label="predict")
    plt.scatter(range(len(pred)), pred, c = "r", label="test")
    plt.savefig("KNN分类.png")
    plt.close()
    # 保存模型
    joblib.dump(knn, "KNN.m")
    return knn
    
    
# 测试选股结果
def test(data, modelfile, modelname):
    model = joblib.load(modelfile)
    pred_return = model.predict(data.iloc[:, 3:21])
    # print(pred_return)
    data["pred_ar"] = pred_return
    # print(data)
    # 排序
    data = data.sort_values(by = "pred_ar", ascending = False)
    # print(data)
    # 取前十个股票作为投资标的
    codes = data.index[0:10].values
    # print(codes)
    return doTest(codes, method = modelname)
    

# 交易策略类，一开始买入然后持有。
class FactorStrategy(bt.Strategy):
    def __init__(self):
        self.p_value = self.broker.getvalue()*0.9/10.0
        self.bOutput = False
        
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
        closeDate = datetime.datetime(2020, 7, 31)
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
        if self.bOutput == False:
            return
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
    # 进行回测,获取各只股票的年化收益率
    data = getReturn(factors)
    # 数据分析
    analysis(data)
    # 进行多元线性回归分析
    multiLineRegress(data)
    # 回测多元线性回归的结果
    mlr = test(data, "LineRegress.m", "多元线性回归")
    # 进行二次多项式回归分析
    quadRegress(data)
    # 回测二次多项式回归结果
    qr = testQuadRegress(data, "QuadRegress.m", "二项式回归")
    # 随机森林回归分析
    RFRegress(data)
    # 回测随机森林回归结果。
    rf = test(data, "RandomForestRegress.m", "随机森林回归")
    # 对数据进行处理，增加标签
    data = addItem(data)
    # 感知机模型
    doPerceptron(data)
    # 用感知机模型进行回测
    pp = test(data.loc[data.res == 1], "Perceptron.m", "感知机模型")
    print(pp)
    # 支持向量机选股
    SVCClassic(data)
    # 用支持向量机回测
    svc = test(data.loc[data.res == 1], "SVC.m", "支持向量机模型")
    print(svc)
    # 逻辑回归选股
    logist(data)
    # 用逻辑回归回测
    log = test(data.loc[data.res == 1], "Logistic.m", "逻辑回归模型")
    print(log)
    # KNN选股
    KNN(data)
    # 用KNN回测
    knn = test(data.loc[data.res == 1], "KNN.m", "KNN模型")
    print(knn)
    # 深度学习进行线性回归
    # DL(data)
    
