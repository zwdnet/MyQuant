# coding:utf-8
# 《量化投资:以python为工具》第三部分


import pandas as pd
import ffn
import matplotlib.pyplot as plt
import datetime
import numpy as np
import math
from scipy import linalg


# 均值类
class MeanVariance():
	def __init__(self, returns):
		self.returns = returns
		
	# 定义最小化方差函数，求解二次规划
	def minVar(self, goalRet):
		covs = np.array(self.returns.cov())
		means = np.array(self.returns.mean())
		L1 = np.append(np.append(covs.swapaxes(0, 1), [means], 0), [np.ones(len(means))], 0).swapaxes(0, 1)
		L2 = list(np.ones(len(means)))
		L2.extend([0, 0])
		L3 = list(means)
		L3.extend([0, 0])
		L4 = np.array([L2, L3])
		L = np.append(L1, L4, 0)
		result = linalg.solve(L, np.append(np.zeros(len(means)), [1, goalRet], 0))
		return (np.array([list(self.returns.columns), results[:-2]]))

	# 最小方差前缘曲线
	def frontierCurve(self):
		goals = [x/500000 for x in range(-100, 4000)]
		variances = list(map(lambda x : self.calVar(self.minVar(x)[1,:].astype(np.float)), goals))
		fig = plt.figure()
		plt.plot(variances, goals)
		plt.savefig("mincurve.png")
		
	# 计算各资产的平均收益率
	def meanRet(self, fracs):
		meanRisky = ffn.to_returns(self.returns).mean()
		assert len(meanRisky) == len(fracs), "meanRisky与fracs长度不相等"
		return (np.sum(np.multiply(meanRisky, np.array(fracs))))
		
	# 计算收益率方差
	def calVar(self, fracs):
		return (np.dot(np.dot(fracs, self.returns.cov()), fracs))
		

if __name__ == "__main__":
	# 计算期间收益率
	stock = pd.read_csv("stockszA.csv", index_col = "Trddt")
	Wanke = stock[stock.Stkcd == 2]
	close = Wanke.Clsprc
	
	print(close.head())
	
	close.index = pd.to_datetime(close.index)
	close.index.name = "Date"
	print(close.head())
	
	lagclose = close.shift(1)
	print(lagclose.head())
	
	Calclose = pd.DataFrame(
	{"close":close,
	 "lagclose":lagclose}
	)
	print(Calclose.head())
	
	# 计算简单单期收益率
	simpleret = (close - lagclose)/lagclose
	simpleret.name = "simpleret"
	print(simpleret.head())
	
	calret = pd.merge(Calclose, pd.DataFrame(simpleret), left_index = True, right_index = True)
	print(calret.head())
	
	# 计算2期简单收益率
	simpleret2 = (close - close.shift(2))/close.shift(2)
	simpleret2.name = "simpleret2"
	calret["simpleret2"] = simpleret2
	print(calret.head())
	
	# 查看1月9日的数据
	print(calret.iloc[5,:])
	
	# 用ffn库计算
	#ffnSimpleret = ffn.to_returns(close)
#	ffnSimpleret.name = "ffnSimpleret"
#	print(ffnSimpleret.head())
	# 计算简单年化收益率
	annualize = (1+simpleret).cumprod()[-1]**(245/311)-1
	print(annualize)
	
	# 绘图
	fig = plt.figure()
	simpleret.plot()
	fig.savefig("return.png")
	fig = plt.figure()
	((1+simpleret).cumprod()-1).plot()
	fig.savefig("cumret.png")
	
	# 计算最大回撤
	r = pd.Series([0, 0.1, -0.1, -0.01, 0.01, 0.02], index = [datetime.date(2015, 7, x) for x in range(3, 9)])
	print(r)
	value = (1+r).cumprod()
	print(value)
	D = value.cummax() - value
	print(D)
	d = D/(D+value)
	print(d)
	MDD = D.max()
	print(MDD)
	mdd = d.max()
	print(mdd)
	
	# 不同相关系数下，投资组合标准差随投资比例变化情况
	def cal_mean(frac):
		return (0.08*frac+0.15*(1-frac))
		
	mean = list(map(cal_mean, [x/50 for x in range(51)]))
	sd_mat = np.array([list(map(lambda x:math.sqrt((x**2)*0.12**2 + ((1-x)**2)*0.25**2 + 2*x*(1-x)*(-1.5+i*0.5)*0.12*0.25), [x/50 for x in range(51)])) for i in range(1, 6)])
	fig = plt.figure()
	plt.plot(sd_mat[0,:], mean, label="-1")
	plt.plot(sd_mat[1,:], mean, label="-0.5")
	plt.plot(sd_mat[2,:], mean, label="0")
	plt.plot(sd_mat[3,:], mean, label="0.5")
	plt.plot(sd_mat[4,:], mean, label="1.0")
	plt.legend(loc = "upper left")
	fig.savefig("risk.png")
	
	# 用Python进行资产配置
	stock = pd.read_table("stock.txt", sep = "\t", index_col = "Trddt")
	fjgs = stock.ix[stock.Stkcd == 600033, "Dretwd"]
	fjgs.name = "fjgs"
	zndl = stock.ix[stock.Stkcd == 600023, "Dretwd"]
	zndl.name = "zndl"
	sykj = stock.ix[stock.Stkcd == 600183, "Dretwd"]
	sykj.name = "sykj"
	hxyh = stock.ix[stock.Stkcd == 600015, "Dretwd"]
	hxyh.name = "hxyh"
	byjc = stock.ix[stock.Stkcd == 600004, "Dretwd"]
	byjc.name = "byjc"
	
	sh_return = pd.concat([byjc, hxyh, sykj, zndl, fjgs], axis = 1)
	print(sh_return.head())
	
	sh_return = sh_return.dropna()
	cumreturn = (1+sh_return).cumprod()
	fig = plt.figure()
	sh_return.plot()
	plt.savefig("sh_return.png")
	cumreturn.plot()
	plt.savefig("sh_cum_return.png")
	
	# 看各股收益的相关性
	print(sh_return.corr())
	# 计算最小方差
	minVar = MeanVariance(sh_return)
	minVar.frontierCurve()
	