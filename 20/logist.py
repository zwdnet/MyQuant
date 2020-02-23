# coding:utf-8
# 鸢尾花数据测试逻辑回归


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris


if __name__ == "__main__":
	iris = load_iris()
	print(iris.data, iris.target)
	
	DD = iris.data
	X = [x[0] for x in DD]
	print(X)
	Y = [x[1] for x in DD]
	print(Y)
	
	fig = plt.figure()
	plt.scatter(X[:50], Y[:50], color = "red", marker = "o", label = "1")
	plt.scatter(X[50:100], Y[50:100], color = "blue", marker = "x", label = "2")
	plt.scatter(X[100:], Y[100:], color = "green", marker = "+", label = "3")
	plt.legend(loc = 2)
	plt.savefig("scatter.png")
	plt.close()
	