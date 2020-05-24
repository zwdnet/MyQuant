# coding:utf-8
# 获取数据并转换成pyalgotrade接受的格式
# 参考https://blog.csdn.net/lawme/article/details/51495349


import tushare as ts
import pandas as pd

from pyalgotrade.bar import Frequency
from pyalgotrade.barfeed.csvfeed import GenericBarFeed
from pyalgotrade.feed import csvfeed
from pyalgotrade import strategy


def downloadData(code):
    data = ts.get_k_data(code, "2019-01-01", "2019-12-31", ktype="5")
    #print(data.head())
#    print(data.info())
    data.to_csv("data.csv")
    df = pd.read_csv("data.csv")
    df2 = pd.DataFrame({'Date Time' : df['date'], 'Open' : df['open'], 'High' : df['high'],'Close' : df['close'], 'Low' : df['low'],'Volume' : df['volume'], 'Adj Close':df['close']})
    # 调整数据为yahoo格式
    dt = df2.pop('Date Time')
    df2.insert(0,'Date Time',dt)
    o = df2.pop('Open')
    df2.insert(1,'Open',o)
    h = df2.pop('High')
    df2.insert(2,'High',h)
    l = df2.pop('Low')
    df2.insert(3,'Low',l)
    c = df2.pop('Close')
    df2.insert(4,'Close',c)
    v = df2.pop('Volume')
    df2.insert(5,'Volume',v)
    # 新格式数据存盘，不保存索引编号
    filename = code+".csv"
    # print(filename, type(filename))
    df2.to_csv(filename, index=False)
    #feed = yahoofeed.Feed()
    #feed.addBarsFromCSV("CB", filename)
    return filename
    
    
# 从csv文件数据导入为pyalgotrade数据源
def buildFeed(code, filename):
    feed = GenericBarFeed(Frequency.MINUTE)
    feed.setDateTimeFormat("%Y-%m-%d %H:%M")
    feed.addBarsFromCSV(code, filename)
    #feed = csvfeed.Feed("Date Time", "%Y-%m-%d %H:%M")
#    feed.addValuesFromCSV(filename)
    #for item in feed:
#        print(item[0], len(item), item[1])
#        print(item.getDateTime())
    return feed
    
    
class MyStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument):
        super(MyStrategy, self).__init__(feed)
        self.__instrument = instrument

    def onBars(self, bars):
        bar = bars[self.__instrument]
        self.info(bar.getClose())
        date = bar.getDateTime()
        print(date.year, date.month, date.day)


if __name__ == "__main__":
    code = "601988"
    filename = downloadData(code)
    # filename = "testdata.csv"
    feed = buildFeed(code, filename)
    myStrategy = MyStrategy(feed, code)
    myStrategy.run()
    