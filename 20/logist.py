# coding:utf-8
# 鸢尾花数据测试逻辑回归


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


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
	
	# 进行逻辑回归
	X = iris.data[:, :2]
	Y = iris.target
	
	# 逻辑回归模型
	lr = LogisticRegression(C = 1e5)
	lr.fit(X, Y)
	
	h = 0.02
	x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
	y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	plt.figure(1, figsize = (8, 6))
	plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
	
	plt.scatter(X[:50, 0], X[:50, 1], color = "red", marker = "o", label = "1")
	plt.scatter(X[50:100, 0], X[50:100, 1], color = "blue", marker = "x", label = "2")
	plt.scatter(X[100:, 0], X[100:, 1], color = "green", marker = "+", label = "3")
	plt.legend(loc = 2)
	plt.savefig("result.png")
	plt.close()
	
	print(lr.score(xx.ravel(), yy.ravel()))
	
