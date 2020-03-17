# coding:utf-8
# kaggle题目泰坦尼克号预测

import tools
import dataHandle as dh
import FeatureEnginner as fe
import modeling
import modelevaluation as me
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = "darkgrid")

import warnings
warnings.filterwarnings('ignore')


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
    me.learnning_curve(rf_parameters, RFC, title, df_all, filename)
