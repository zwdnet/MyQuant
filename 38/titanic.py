# coding:utf-8
# kaggle题目泰坦尼克号预测

import tools
import dataHandle as dh
import FeatureEnginner as fe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = "darkgrid")

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler

SEED = 42
            
        
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
    # 划分数据
    df_train, df_test = tools.divide_df(df_all)
    X_train = StandardScaler().fit_transform(df_train.drop(["Survived"], axis = 1))
    y_train = df_train["Survived"].values
    X_test = StandardScaler().fit_transform(df_test)
    
    print('X_train shape: {}'.format(X_train.shape))
    print('y_train shape: {}'.format(y_train.shape))
    print('X_test shape: {}'.format(X_test.shape))
