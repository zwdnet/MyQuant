# coding:utf-8
# 回测程序的单元测试


import sys
sys.path.append("../")
import trade
import datetime


# 测试读取数据函数
def test_get_data():
    start = "2018-01-01"
    end = "2020-07-05"
    df_300 = trade.getData("510300", start, end)
    assert sorted(df_300.columns) == sorted(['open','high','low','close','volume','openinterest'])