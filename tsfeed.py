# coding:utf-8
"""
  tsfeed 用于将tushare的数据转换为
  pyalgotrade可以接受的格式
  参考:https://my.oschina.net/lucasli/blog/1791018
"""


from pyalgotrade.barfeed import membf
from pyalgotrade import bar
import tushare as ts
import datetime
import pandas as pd


def parse_date(date):
	year = int(date[0:4])
	month = int(date[5:7])
	day = int(date[8:10])
	d = datetime.datetime(year, month, day)
	
	if len(date) > 10:
		h = int(date[11:13])
		m = int(date[14:16])
		t = datetime.time(h, m)
		ret = datetime.combine(d, t)
	else:
		ret = d
	
	return ret
	

"""将tushare取得的数据转换为pyalgotrade可以接受的数据源，可以直接输股票代码和起止时间，也可以输入csv文件。"""		
class Feed(membf.BarFeed):
	def __init__(self, frequency = bar.Frequency.DAY, maxLen = None):
		super(Feed, self).__init__(frequency, maxLen)
		
	def rowParser(self, ds, frequency = bar.Frequency.DAY):
		dt = parse_date(ds["date"])
		open = float(ds["open"])
		close = float(ds["close"])
		high = float(ds["high"])
		low = float(ds["low"])
		volume = float(ds["volume"])
		return bar.BasicBar(dt, open, high, low, close, volume, None, frequency)
		
	def barsHaveAdjClose(self):
		return False
	
	# 辅助函数，完成数据整合		
	def makeBars(self, instrument, ds, frequency):
		bars_ = []
		for i in ds.index:
			bar_ = self.rowParser(ds.loc[i], frequency)
			bars_.append(bar_)
		self.addBarsFromSequence(instrument, bars_)
	
	# 获取数据周期		
	def makeFrequency(self, ktype):
		frequency = bar.Frequency.DAY
		if ktype == "D":
			frequency = bar.Frequency.DAY
		elif ktype == "W":
			frequency = bar.Frequency.WEEK
		elif ktype == "M":
			frequency = bar.Frequency.MONTH
		elif ktpye == "5" or ktpye == "15" or ktpye == "30" or ktpye == "60":
			frequency = bar.Frequency.MINUTE
		
		return frequency
	
	# 从股票代码建立数据源		
	def addBarsFromCode(self, code, start, end, ktype = "D", index = False):
		frequency = self.makeFrequency(ktype)
		ds = ts.get_k_data(code = code, start = start, end = end, ktype = ktype, index = index)
		self.makeBars(code, ds, frequency)
	
	# 从csv文件建立数据源		
	def addBarsFromCsv(self, code, path, start, end, ktype = "D"):
		frequency = self.makeFrequency(ktype)
		ds = pd.read_csv(path)
		self.makeBars(code, ds, frequency)
		
	
if __name__ == "__main__":
	pass