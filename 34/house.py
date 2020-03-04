# coding:utf-8
# 岭回归实例化，波士顿房价预测


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston


if __name__ == "__main__":
    # 加载数据
    data = load_boston()
    boston = pd.DataFrame(data.data, columns = data.feature_names)
    y = data.target
    boston["PRICE"] = y
    print(boston.head())
    print(boston.info())
