# coding:utf-8
# 神经网络实现手写识别


import numpy as np
from os import listdir
# from sklearn.neural_network import MLPclassifier


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
    fileList = listdir(path)
    numFiles = len(fileList)
    dataSet = np.zeros([numFiles, 1024], int)
    hwLabels = np.zeros([numFiles, 10])
    for i in range(numFiles):
        filePath = fileList[i]
        digit = int(filePath.split("_")[0])
        hwLabels[i][digit] = 1.0
        dataSet[i] = img2vector(path + '/' + filePath)
    return dataSet, hwLabels


if __name__ == "__name__":
    print("hello")
    train_dataSet, train_hwLabels = readDataSet('./data/pendigits-orig.tes.Z')
    print(len(train_dataSet))
    print(train_hwLabels)
    