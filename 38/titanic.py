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


# 画ROC曲线
def plot_roc_curve(fprs, tprs):
    
    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=(15, 15))
   
   # 为每次折叠测试画ROC曲线并计算AUC值
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs), 1):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC Fold {} (AUC = {:.3f})'.format(i, roc_auc))
        
    # 为随机猜测画ROC图
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8, label='Random Guessing')
    
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    # 画平均ROC值
    ax.plot(mean_fpr, mean_tpr, color='b', label='Mean ROC (AUC = {:.3f} $\pm$ {:.3f})'.format(mean_auc, std_auc), lw=2, alpha=0.8)
    
    # 画平均ROC的标准差
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label='$\pm$ 1 std. dev.')
    
    ax.set_xlabel('False Positive Rate', size=15, labelpad=20)
    ax.set_ylabel('True Positive Rate', size=15, labelpad=20)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    ax.set_title('ROC Curves of Folds', size=20, y=1.02)
    ax.legend(loc='lower right', prop={'size': 13})
    
    plt.savefig("ROC.png")
            
        
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
    
    # 画图看看
    importances['Mean_Importance'] = importances.mean(axis=1)
    importances.sort_values(by='Mean_Importance', inplace=True, ascending=False)
    
    plt.figure(figsize=(15, 20))
    sns.barplot(x='Mean_Importance', y=importances.index, data=importances)
    
    plt.xlabel('')
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    plt.title('Random Forest Classifier Mean Feature Importance Between Folds', size=15)
    
    plt.savefig("RandomForest.png")
    plt.close()
    
    # 画ROC曲线
    plot_roc_curve(fprs, tprs)

