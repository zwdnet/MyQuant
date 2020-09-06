# coding:utf-8
# 用深度学习进行非线性回归，多因子选股


import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F


# 训练模型
def fit(num_epochs, model, loss_fn, opt):
    for epoch in range(num_epochs):
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
            
        if (epoch+1) % 10 == 0:
            print("Epoch [{}/{}], Loss:{:.4f}".format(epoch+1, num_epochs, loss.item()))


def DL(data):
    x = data.iloc[:, 3:21]
    y = data.iloc[:, 22]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 631)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)
    
    # 定义数据集
    train_ds = TensorDataset(inputs, targets)
    print(train_ds[0:18])
    # 定义data loader
    batch_size = 5
    train_dl = DataLoader(train_ds, batch_size, shuffle = True)
    for xb, yb in train_dl:
        print(xb)
        print(yb)
        break
    # 定义模型
    model = nn.Linear(3, 2)
    print(model.weight)
    print(model.bias)
    # 输出参数
    # print(list(model.parameters))
    # 进行预测
    # preds = model(inputs.float())
    # print(preds)
    # 损失函数
    loss_fn = F.mse_loss
    loss = loss_fn(model(inputs), targets)
    print(loss)
    # 优化器
    opt = torch.optim.SGD(model.parameters(), lr = 1e-5)
    
    fit(100, model, loss_fn, opt)
    preds = model(inputs)
    print(preds)
    print(targets)
    
