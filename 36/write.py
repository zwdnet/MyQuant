# coding:utf-8
# 神经网络实现手写识别


import numpy as np
from os import listdir
from sklearn.neural_network import MLPClassifier


# 加载训练数据
def img2vector(filename):
    retMat = np.zeros([1024], int)
    fr = open(filename)
    lines = fr.readlines()
    for i in range(32):
        for j in range(32):
            retMat[i*32+j] = lines[i][j]
    return retMat
    
    
# 加载训练数据集，并将标签转化为one-hot向量
def readDataSet(Path):
    fileList = listdir(Path)
    numFiles = len(fileList)
    dataSet = np.zeros([numFiles, 1024], int)
    hwLabels = np.zeros([numFiles, 10])
    for i in range(numFiles):
        filePath = fileList[i]
        digit = int(filePath.split("_")[0])
        hwLabels[i][digit] = 1.0
        dataSet[i] = img2vector(path + '/' + filePath)
    return dataSet, hwLabels


if __name__ == "__main__":
    # 读取数据，照https://github.com/JayZhuCoding/simple_neural_network/blob/master/Neural.py来
    dataset = np.genfromtxt("./data/pendigits.tra", delimiter= ",")
    m, n = dataset.shape
    X = dataset[:,:n-1]
    y = dataset[:,n-1]
    X /= X.max()
    
    # 建立神经网络
    clf = MLPClassifier(hidden_layer_sizes = (100,), activation = "logistic", solver = "adam", learning_rate_init = 0.0001, max_iter = 2000)
    # 训练神经网络
    clf.fit(X, y)
    # 载入测试集
    dataset_test = np.genfromtxt("./data/pendigits.tes", delimiter= ",")
    m_test, n_test = dataset_test.shape
    X_test = dataset_test[:,:n_test-1]
    y_test = dataset[:,n_test-1]
    X_test /= X_test.max()
    # 测试
    res = clf.predict(X_test)
    error_sum = 0
    num = len(X_test)
    for i in range(num):
        if np.sum(res[i] == y_test[i]) < 10:
            error_sum += 1
    print(num, error_sum, error_sum/float(num))
    
