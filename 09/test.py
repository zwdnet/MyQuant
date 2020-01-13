# coding:utf-8
# 《量化投资:以python为工具》第三部分


import pandas as pd
import ffn


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
	ffnSimpleret = ffn.to_returns(close)
	ffnSimpleret.name = "ffnSimpleret"
	print(ffnSimpleret.head())
	