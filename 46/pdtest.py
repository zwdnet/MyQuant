# coding :utf-8
# pandas基本操作


import pandas as pd
import numpy as np
from datetime import datetime


if __name__ == "__main__":
    df_record = pd.read_csv("etfdata.csv")
    df = pd.read_csv("510300.csv")
    df_record.成交日期 = pd.to_datetime(df_record.成交日期, format = "%Y%m%d")
    df.date = pd.to_datetime(df.date, format = "%Y-%m-%d")
    df_record.index = df_record.成交日期
    df.index = df.date
    df = df[['open','high','low','close','volume']]
    df_record = df_record[["成交日期", "证券名称","成交量","成交均价","成交金额","手续费","发生金额"]]
    print(df_record.head())
    print(df.head())

    x = df_record.loc["2018-3-13", :]
    print(x)
    y = df.index.strftime("%Y-%m-%d")
    for d in y:
        x = df_record.loc[d, ["成交日期", "证券名称","成交量","成交均价","成交金额","手续费","发生金额"]]
        if len(x) != 0:
            print(x)
    