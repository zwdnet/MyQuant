# coding:utf-8
# 使用tushare获取数据，保存到csv文件


import tushare as ts
import pandas as pd


# 获取历史数据
def GetHistroyData(code, startTime, endTime):
	data = ts.get_k_data(code, start = startTime, end = endTime)
	return data


if __name__ == "__main__":
	PinganData = GetHistroyData("000001", "2016-01-01", "2016-12-31")
	print(len(PinganData))
	HS300Data = GetHistroyData("hs300",
	"2016-01-01", "2016-12-31")
	print(len(HS300Data))
	PinganData.to_csv("PA_his.csv")
	HS300Data.to_csv("HS300_his.csv")
	