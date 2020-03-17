# coding:utf-8
# kaggle题目泰坦尼克号预测
# 工具函数


import numpy as np
import pandas as pd


# 将训练数据和训练数据合并到一起
def concat_df(train_data, test_data):
    return pd.concat([train_data, test_data], sort = True).reset_index(drop = True)
    
    
# 将数据集重新分割为训练集和测试集
def divide_df(all_data):
    return all_data.loc[:890], all_data.loc[891:].drop(["Survived"], axis = 1)
    
    
# 加载数据
def load_data():
    df_train = pd.read_csv("./data/train.csv")
    df_test = pd.read_csv("./data/test.csv")
    df_all = concat_df(df_train, df_test)
    
    df_train.name = "Training Set"
    df_test.name = "Test Set"
    df_all.name = "All Set"
    
    dfs = [df_train, df_test]
    
    print("训练样本量 = {}".format(df_train.shape[0]))
    print("测试样本量 = {}".format(df_test.shape[0]))
    print("训练中的X的形状 = {}".format(df_train.shape))
    print("训练中的y的形状 = {}".format(df_train["Survived"].shape[0]))
    print("测试中的X的形状 = {}".format(df_test.shape))
    print("测试中的y的形状 = {}".format(df_test.shape[0]))
    print(df_train.columns)
    print(df_test.columns)
    
    # 查看数据情况
    print(df_train.info())
    print(df_train.sample(3))
    print(df_test.info())
    print(df_test.sample(3))
    
    return df_all
    
    
# 创建提交kaggle的结果文件
def Submission(model, df_test, filename):
    submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])
    submission_df['PassengerId'] = df_test['PassengerId']
    submission_df['Survived'] = model.predict(df_test.drop('PassengerId', axis=1).copy())
    submission_df.to_csv(filename, header=True, index=False)
    print(submission_df.head(10))
    