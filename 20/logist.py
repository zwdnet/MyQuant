# coding:utf-8
# 鸢尾花数据测试逻辑回归


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris


if __name__ == "__main__":
	iris = load_iris()
	print(iris.head())
	