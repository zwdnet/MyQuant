# coding:utf-8
# 用自己编的数据计算回测指标
import pandas as pd
import numpy as np
import empyrical as ep
import statsmodels.api as sm
from scipy import stats


if __name__ == "__main__":
	# 从文件中读取数据
	test_df = pd.read_csv("test.csv", index_col = "Date")
	print(test_df)
	base_df = pd.read_csv("base.csv", index_col = "Date")
	print(base_df)
	# 提取收盘价信息
	test_close = test_df["Close"]
	base_close = base_df[["Close"]]
	# test_close.name = "Close"
	# base_close.name = "Close"
	print(test_close, base_close)
	# 计算每日收益率
	# 初始投资
	cash_test = 2.5
	cash_base = 2.5
	# 每期市值
	position_test = []
	position_base = []
	print(test_close.values[0], base_close.values[0])
	for i in range(len(test_close)):
		if i == 0:
			position_test.append(cash_test)
			position_base.append(cash_base)
			continue
		elif i == 1:
			cash_test = cash_test - test_close[0]
			cash_base = cash_base - base_close.values[0][0]
		if cash_test <= 0 or cash_base <= 0:
			print("现金不足，退出")
		position_test.append(cash_test + test_close[i-1])
		position_base.append(cash_base + base_close.values[i-1][0])
		
	print(position_test, position_base)
	test_return = []
	base_return = []
	test_return.append(0.0)
	base_return.append(0.0)
	for i in range(1, len(position_test)):
		print(i, position_test[i], position_test[i-1])
		test_return.append((position_test[i] - position_test[i-1])/position_test[i-1])
		base_return.append((position_base[i] - position_base[i-1])/position_base[i-1])
	# test_return.values[0] = 0.0
	# base_return.values[0] = 0.0
	print(test_return)
	print(base_return)
	# 计算年化收益率
	np_test_return = np.array(test_return)
	np_base_return = np.array(base_return)
	annret_test = (1+np_test_return).cumprod()[-1]**(245/311) - 1
	annret_base = (1+np_base_return).cumprod()[-1]**(245/311) - 1
	print(annret_test, annret_base)
	print("empyrical")
	annret_test_ep = ep.annual_return(np_test_return)
	annret_base_ep = ep.annual_return(np_base_return)
	print(annret_test_ep, annret_base_ep)
	
	# 衡量风险
	# 标准差
	print(np_test_return.std(), np_base_return.std())
	# 最大回撤
	print(ep.max_drawdown(np_test_return), ep.max_drawdown(np_base_return))
	# 计算αβ值
	# 先将两个收益率合并到一起
	Ret = pd.merge(pd.DataFrame(base_return), pd.DataFrame(test_return),  left_index = True, right_index = True, how = "inner")
	print(Ret)
	# 计算无风险收益
	rf = 1.036**(1/360) - 1.0
	print(rf)
	# 计算股票超额收益率和市场风险溢酬
	Eret = Ret - rf
	print(Eret)
	# 接下来进行拟合
	model = sm.OLS(np_test_return, sm.add_constant(np_base_return))
	result = model.fit()
	print(result.summary())
	print("empyrical")
	alpha, beta = ep.alpha_beta(np_test_return, np_base_return, 0.036)
	print(alpha, beta)
	# 另一种方法
	x = []
	y = []
	print(test_return)
	for i in range(10):
		x.append(base_return[i])
		y.append(test_return[i])
	b, a, r_value, p_value, std_err = stats.linregress(x, y)
	print(a, b)
	
	# 计算夏普比率
	sharpe = (np_test_return.mean() - 0.03)/np_test_return.std()*np.sqrt(252)
	print(sharpe)
	print(ep.sharpe_ratio(np_test_return, risk_free = 0.03))
	