# coding:utf-8
# python基础知识补遗


import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters


if __name__ == "__main__":
	x = complex(2, 5)
	print(x)
	y = 3-6j
	print(y)
	z = 3-6j
	print(id(y), id(z))
	# 改变参数
	def testChange(x, y):
		x[0] = "A"
		y = 7
		
	x = ["a", "b", "c", "d"]
	y = 6
	testChange(x, y)
	print(x, y)
	# 不定参数
	def manyCan(*arg):
		sum = 0
		for i in arg:
			sum = sum+i
		return sum
		
	print(manyCan(1,2,3))
	
	# 匿名函数
	greeting = lambda : print("hello")
	greeting()
	
	# 字典测试
	dictest = {"High":5, "Low":1, "Close":3}
	print(dictest)
	for key in dictest.keys():
		print(key)
		print(dictest[key])
			
	# numpy的array相关操作
	array1 = np.array(range(6))
	print(array1)
	print(array1.shape)
	array1.shape = 2,3
	print(array1)
	array2 = array1.reshape(3,2)
	print(array2)
	array1[1,2] = 88
	print(array1, array2)
	array3 = np.array([[1,2,3], 
									[4,5,6],
									[7,8,9]])
	print(array3)
	array4 = np.arange(13, 1, -1)
	print(array4)
	array4.shape = 2,2,3
	print(array4)
	array5 = array4.reshape(3,2,2)
	print(array5)
	array6 = np.linspace(1, 12, 12)
	print(array6)
	print(array6.dtype)
	array7 = np.linspace(1, 12, 12, dtype = int)
	print(array7)
	
	# pandas测试
	s1 = pd.Series()
	print(s1)
	s2 = pd.Series([1, 3, 5, 7, 9], index = ["a", "b", "c", "d", "e"])
	print(s2)
	print(s2.index, s2.values)
	s2["f"] = 11
	print(s2)
	np.random.seed(54321)
	s3 = pd.Series(np.random.randn(5))
	print(s3)
	date = datetime(2016,1,1)
	date = pd.Timestamp(date)
	ts = pd.Series(1, index = [date])
	print(date)
	print(ts)
	dates = [datetime(2016, 1, 1), datetime(2016, 1, 2), datetime(2016, 1, 3)]
	ts = pd.Series([1, 2, 3], index = dates)
	print(ts)
	print(ts.shift(1))
	date = ["2016-01-01", "2016-01-02", "2016-01-03", "2016-01-04", "2016-01-05", "2016-01-06"]
	dates = pd.to_datetime(date)
	print(dates)
	df = pd.DataFrame(np.random.randn(6, 4), index = dates, columns = list("ABCD"))
	print(df)
	print(df.head(3))
	print(df.tail(3))
	print(df.columns)
	print(df.index)
	print(df.values)
	print(df.describe())
	print(df[1:3])
	print(df["A"])
	print(df[df["A"]>0])
	print(df.loc[:,"A"])
	print(df.loc[:, "A":"C"])
	print(df.loc[dates[0:2], "A":"C"])
	print(df.loc[dates[0], "A"])
	print(df.iloc[2])
	print(df.iloc[:, 2])
	print(df.iloc[1:4, 2:3])
	print(df.iloc[[1,4], [2,3]])
	print(df.iloc[3,3])
	print(df.iat[3,3])
	print(df.ix[2:5])
	print(df.T)
	print(df.rank(axis = 0))
	
	# matplotlib
	register_matplotlib_converters()
	ChinaBank = pd.read_csv("ChinaBank.csv", index_col = "Date")
	ChinaBank = ChinaBank.iloc[:, 1:]
	print(ChinaBank.head())
	ChinaBank.index = pd.to_datetime(ChinaBank.index)
	Close = ChinaBank.Close
	fig = plt.figure()
	plt.plot(Close["2014"])
	fig.savefig("close2014.png")
	fig = plt.figure()
	plt.hist(Close["2014"], bins = 12)
	fig.savefig("close2014hist.png")
	