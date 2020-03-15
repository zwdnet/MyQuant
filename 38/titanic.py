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
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, roc_curve

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

    single_best_model = RFC(criterion = "gini", n_estimators = 1100, max_depth = 5, min_samples_split=4, min_samples_leaf=5, max_features='auto', oob_score=True, random_state=SEED, n_jobs=-1, verbose=1)
    leaderboard_model = RFC(criterion = "gini", n_estimators = 1750, max_depth = 7, min_samples_split=6, min_samples_leaf=6, max_features='auto', oob_score=True, random_state=SEED, n_jobs=-1, verbose=1)
    N = 5
    oob = 0
    probs = pd.DataFrame(np.zeros((len(X_test), N*2)), columns = ['Fold_{}_Prob_{}'.format(i, j) for i in range(1, N + 1) for j in range(2)])
    df_temp = df_all.drop(["Survived"], axis = 1)
    importances = pd.DataFrame(np.zeros((X_train.shape[1], N)), columns=['Fold_{}'.format(i) for i in range(1, N + 1)], index=df_temp.columns)
    fprs, tprs, scores = [], [], []
    skf = StratifiedKFold(n_splits=N, random_state=N, shuffle=True)
    
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print('Fold {}\n'.format(fold))
        
        # 模型拟合
        leaderboard_model.fit(X_train[trn_idx], y_train[trn_idx])
        
        # 计算训练的AUC分数
        trn_fpr, trn_tpr, trn_thresholds = roc_curve(y_train[trn_idx], leaderboard_model.predict_proba(X_train[trn_idx])[:, 1])
        trn_auc_score = auc(trn_fpr, trn_tpr)
        # 计算检验的AUC分数
        val_fpr, val_tpr, val_thresholds = roc_curve(y_train[val_idx], leaderboard_model.predict_proba(X_train[val_idx])[:, 1])
        val_auc_score = auc(val_fpr, val_tpr)
        
        scores.append((trn_auc_score, val_auc_score))
        fprs.append(val_fpr)
        tprs.append(val_tpr)
        
        # X_test概率
        probs.loc[:, 'Fold_{}_Prob_0'.format(fold)] = leaderboard_model.predict_proba(X_test)[:, 0]
        probs.loc[:, 'Fold_{}_Prob_1'.format(fold)] = leaderboard_model.predict_proba(X_test)[:, 1]
        importances.iloc[:, fold - 1] = leaderboard_model.feature_importances_
        
        oob += leaderboard_model.oob_score_ / N
        print('Fold {} OOB Score: {}\n'.format(fold, leaderboard_model.oob_score_))
    
    print('Average OOB Score: {}'.format(oob))
    
    
    
