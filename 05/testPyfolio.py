# coding:utf-8
# 测试pyfolio

import pyfolio as pf
import pandas as pd
import matplotlib.pyplot as plt


return_ser = pd.read_csv("return_ser.csv")
return_ser["date"] = pd.to_datetime(return_ser["date"])
return_ser.set_index("date", inplace = True)
pf.create_returns_tear_sheet(return_ser["return"])
