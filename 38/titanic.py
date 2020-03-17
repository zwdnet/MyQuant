# coding:utf-8
# kaggle题目泰坦尼克号预测

import tools
import dataHandle as dh
import FeatureEnginner as fe
import modeling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = "darkgrid")

import warnings
warnings.filterwarnings('ignore')


from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.preprocessing import StandardScaler


SEED = 42


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
def learnning_curve(parameters, model, title, df_all, filename):
    df_train, df_test = tools.divide_df(df_all)
    X_train = StandardScaler().fit_transform(df_train.drop(["Survived"], axis = 1))
    y_train = df_train["Survived"].values
    plt = plot_learning_curve(model(**parameters), title, X_train, y_train, cv=None,  n_jobs=-1, train_sizes=[50, 100, 150, 200, 250, 350, 400, 450, 500])
    plt.savefig(filename)


if __name__ == "__main__":
    # 载入数据
    df_all = tools.load_data()
    
    # 数据处理
    # ①填充缺失值
    df_all = dh.fillna_data(df_all)
    # ②数据探索
    dh.exploratory_data(df_all)

    print(df_all.head())
    
    # ③特征工程
    df_all = fe.feature_engineer(df_all)
    
    # ④建模
    modeling.model(df_all)
    
    # ⑤模型评估
    rf_parameters = {"criterion":"gini", "n_estimators":1750, "max_depth":7, "min_samples_split":6, "min_samples_leaf":6, "max_features":'auto', "oob_score":True, "random_state":SEED, "n_jobs":-1, "verbose":1}
    title = "RandomForest"
    filename = "learningCurve.png"
    learnning_curve(rf_parameters, RFC, title, df_all, filename)
