# coding:utf-8
# kaggle题目泰坦尼克号预测
# 模型评估


import tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = "darkgrid")

from sklearn.model_selection import learning_curve, validation_curve, cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.preprocessing import StandardScaler


SEED = 42


# 模型评分
def ModelTest(Model, X_train, Y_train):
    Model.fit(X_train, Y_train)
    # 对模型评分
    acc_result = round(Model.score(X_train, Y_train)*100, 2)
    return acc_result


# 绘制模型的学习曲线
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,train_sizes=np.linspace(.1, 1.0, 5),verbose=0):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv,n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes,test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# 学习曲线
# parameters 模型参数
# model 模型
# df_all 数据
# filename 输出图片的文件名
def learnning_curve(model, title, df_all, filename):
    df_train, df_test = tools.divide_df(df_all)
    X_train = StandardScaler().fit_transform(df_train.drop(["Survived"], axis = 1))
    y_train = df_train["Survived"].values
    plt = plot_learning_curve(model, title, X_train, y_train, cv=None,  n_jobs=-1, train_sizes=[50, 100, 150, 200, 250, 350, 400, 450, 500])
    plt.savefig(filename)
    

# 对模型进行交叉验证
def cross_val(model, X, Y, cv=5):
    scores = cross_val_score(model, X, Y, cv=cv)
    score = scores.mean()
    return score
