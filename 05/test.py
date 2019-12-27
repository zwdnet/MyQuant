# coding:utf-8
# 用知乎上的文章https://zhuanlan.zhihu.com/p/55425806检验计算回测指标是否正确

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
from scipy import stats


def get_data(code, start_date = "2014-01-01", end_date = "2018-12-31"):
	df = ts.get_k_data(code, start = start_date, end = end_date)
	df.index = pd.to_datetime(df.date)
	return df.close


if __name__ == "__main__":
	stocks={'510300':'沪深300etf','600519':'贵州茅台','601398':'工商银行','601318':'中国平安'}
	df = pd.DataFrame()
	for code, name in stocks.items():
		df[name] = get_data(code)
	
	df_new = df / df.iloc[0]
	# df_new = df
	
	# 区间累积收益率(绝对收益率)
	total_ret = df_new.iloc[-1] - 1.0
	TR=pd.DataFrame(total_ret.values,columns=['累计收益率'],index=total_ret.index)
	print(TR)
	
	# 年化收益率
	annual_ret=pow(1+total_ret,250/len(df_new))-1
	AR=pd.DataFrame(annual_ret.values,columns=['年化收益率'],index=annual_ret.index)
	print(AR)
	
	# 最大回撤
	def max_drawdown(df):
		md = ((df.cummax() - df)/df.cummax()).max()
		return round(md, 4)
		
	md = {}
	for code, name in stocks.items():
		md[name] = max_drawdown(df[name])
	MD = pd.DataFrame(md, index = ["最大回撤"]).T
	print(MD)
	
	# 计算alpha和beta
	rets = (df.fillna(method = "pad")).apply(lambda x:x/x.shift(1)-1)[1:]
	print(rets.head())
	
	x = rets.iloc[:, 0].values
	y = rets.iloc[:, 1:].values
	AB = pd.DataFrame()
	alpha = []
	beta = []
	for i in range(3):
		b, a, r_value, p_value, std_err = stats.linregress(x, y[:, i])
		alpha.append(round(a*250, 3))
		beta.append(round(b, 3))
	AB["alpha"] = alpha
	AB["beta"] = beta
	AB.index = rets.columns[1:]
	print(AB)
	
	# 公式法计算αβ
	beta1=rets[['沪深300etf','贵州茅台']].cov().iat[0,1]/rets['沪深300etf'].var()
	beta2=rets[['沪深300etf','工商银行']].cov().iat[0,1]/rets['沪深300etf'].var()
	beta3=rets[['沪深300etf','中国平安']].cov().iat[0,1]/rets['沪深300etf'].var()
	
	print(f'贵州茅台beta:{round(beta1,3)}')
	print(f'工商银行beta:{round(beta2,3)}')
	print(f'中国平安beta:{round(beta3,3)}')
	
	alpha1=(annual_ret[1]-annual_ret[0]*beta1)
	alpha2=(annual_ret[2]-annual_ret[0]*beta2)
	alpha3=(annual_ret[3]-annual_ret[0]*beta3)
	print(f'贵州茅台alpha:{round(alpha1,3)}')
	print(f'工商银行alpha:{round(alpha2,3)}')
	print(f'中国平安alpha:{round(alpha3,3)}')
	
	# 夏普比率和信息比率
	# 无风险收益为3%
	exReturn = rets - 0.03/250
	# 计算夏普比率
	sharperatio = np.sqrt(len(exReturn))*exReturn.mean()/exReturn.std()
	SHR = pd.DataFrame(sharperatio, columns = ["夏普比率"])
	print(SHR)
	
	# 计算信息比率，以指数为基准收益
	ex_return = pd.DataFrame()
	ex_return['贵州茅台']=rets.iloc[:,1]-rets.iloc[:,0]
	ex_return['工商银行']=rets.iloc[:,2]-rets.iloc[:,0]
	ex_return['中国平安']=rets.iloc[:,3]-rets.iloc[:,0]
	
	information = np.sqrt(len(ex_return))*ex_return.mean()/ex_return.std()
	INR = pd.DataFrame(information, columns = ["信息比率"])
	print(INR)
	
	# 合并到一起
	indicators = pd.concat([TR, AR, MD, AB, SHR, INR], axis = 1, join = "outer", sort = False)
	print(indicators.round(3))
	