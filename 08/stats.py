# coding:utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.stats.anova as anova
from statsmodels.formula.api import ols
import statsmodels.api as sm


if __name__ == "__main__":
	returns = pd.read_csv("retdata.csv")
	# 数据位置
	print(returns.zglt.mean(), returns.zglt.median(), returns.zglt.mode())
	print(returns.pfyh.mean(), returns.pfyh.median(), returns.pfyh.mode())
	# 数据离散度
	print(returns.zglt.max() - returns.zglt.min(), returns.zglt.mad(), returns.zglt.var(), returns.zglt.std())
	
	# 使用choice
	numSize = 100
	RandomNumber = np.random.choice([1,2,3,4,5], size = numSize, replace = True,  p = [0.1, 0.1, 0.3, 0.3, 0.2])
	print(pd.Series(RandomNumber).value_counts())
	print(pd.Series(RandomNumber).value_counts()/numSize)
	# 连续型随机变量
	HSRet300 = pd.read_csv("return300.csv")
	print(HSRet300.head(2))
	
	fig = plt.figure()
	density = stats.kde.gaussian_kde(HSRet300.iloc[:, 1])
	bins = np.arange(-5, 5, 0.02)
	plt.subplot(211)
	plt.plot(bins, density(bins))
	plt.subplot(212)
	plt.plot(bins, density(bins).cumsum())
	fig.savefig("result.png")
	
	# 二项式分布
	print(np.random.binomial(100, 0.5, 20))
	print(stats.binom.pmf(20, 100, 0.5))
	
	ret = HSRet300.iloc[:, 1]
	print(ret.head(3))
	p = len(ret[ret > 0])/len(ret)
	print(p)
	prob = stats.binom.pmf(6, 10, p)
	print(prob)
	
	# 正态分布
	Norm = np.random.normal(size = 5)
	print(Norm)
	print(stats.norm.pdf(Norm))
	print(stats.norm.cdf(Norm))
	
	# 卡方分布
	x = np.arange(0, 5, 0.002)
	fig = plt.figure()
	plt.plot(x, stats.chi.pdf(x, 3))
	fig.savefig("X.png")
	
	# t分布
	x = np.arange(-4, 4.004, 0.004)
	fig = plt.figure()
	plt.plot(x, stats.norm.pdf(x), label = "Normal")
	plt.plot(x, stats.t.pdf(x, 5), label = "df = 5")
	plt.plot(x, stats.t.pdf(x, 30), label = "df = 30")
	plt.legend()
	fig.savefig("t.png")
	
	# F分布
	x = np.arange(0, 5, 0.002)
	fig = plt.figure()
	plt.plot(x, stats.f.pdf(x, 4, 40))
	fig.savefig("f.png")
	
	# 股指的线性相关性
	TRD_Index = pd.read_table("TRD_Index.txt", sep = "\t")
	SHindex = TRD_Index[TRD_Index.Indexcd == 1]
	print(SHindex.head())
	SZindex = TRD_Index[TRD_Index.Indexcd == 399106]
	print(SZindex.head())
	
	fig = plt.figure()
	plt.scatter(SHindex.Retindex, SZindex.Retindex)
	fig.savefig("index.png")
	
	SZindex.index = SHindex.index
	cor = SZindex.Retindex.corr(SHindex.Retindex)
	print(cor)
	
	# 区间估计
	x = [10.1, 10, 9.8, 10.5, 9.7, 10.1, 9.9, 10.2, 10.3, 9.9]
	xp = stats.t.interval(0.95, len(x)-1, np.mean(x), stats.sem(x))
	print(xp)
	
	# 单样本t检验
	SHRet = SHindex.Retindex
	print(stats.ttest_1samp(SHRet, 0))
	# 独立样本t检验
	SZRet = SZindex.Retindex
	print(stats.ttest_ind(SHRet, SZRet))
	# 配对t检验
	print(stats.ttest_rel(SHRet, SZRet))
	
	year_return = pd.read_csv("TRD_Year.csv", encoding = "gbk")
	print(year_return.head())
	
	model = ols("Return ~ C(Industry)", data = year_return.dropna()).fit()
	table1 = anova.anova_lm(model)
	print(table1)
	
	# 线性回归
	SZRet.index = SHRet.index
	model = sm.OLS(SHRet, sm.add_constant(SZRet)).fit()
	print(model.summary())
	fig = plt.figure()
	plt.scatter(model.fittedvalues, model.resid)
	fig.savefig("line_return.png")
	fig = plt.figure()
	fig = sm.qqplot(model.resid_pearson, stats.norm, line = "45")
	fig.savefig("test_norm.png")
	