# coding:utf-8
# kaggle的房价预测题
# 参考https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style("darkgrid")
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
from scipy import stats
from scipy.stats import norm, skew
pd.set_option("display.float_format", lambda x:"{:.3f}".format(x))


from sklearn.preprocessing import LabelEncoder


if __name__ == "__main__":
    # 载入数据
    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")
    
    print(train.head(5))
    print(test.head(5))
    print(train.shape)
    print(test.shape)
    
    # 保存ID值
    train_ID = train["Id"]
    test_ID = test["Id"]
    # 从数据中丢弃"Id"列
    train.drop("Id", axis = 1, inplace = True)
    test.drop("Id", axis = 1, inplace = True)
    print(train.shape)
    print(test.shape)
    
    # 数据处理
    # 探索异常值
    fig, ax = plt.subplots()
    ax.scatter(x = train["GrLivArea"], y = train["SalePrice"])
    plt.ylabel("SalePrice", fontsize = 13)
    plt.xlabel("GrLivArea", fontsize = 13)
    plt.savefig("outliers.png")
    # 删除异常值
    train = train.drop(train[(train["GrLivArea"] > 4000) & (train['SalePrice']<300000)].index)
    fig, ax = plt.subplots()
    ax.scatter(x = train["GrLivArea"], y = train["SalePrice"])
    plt.ylabel("SalePrice", fontsize = 13)
    plt.xlabel("GrLivArea", fontsize = 13)
    plt.savefig("outliers_afterdel.png")
    plt.close()
    # 研究目标变量SalePrice
    plt.figure()
    sns.distplot(train["SalePrice"], fit = norm)
    (mu, sigma) = norm.fit(train["SalePrice"])
    print("mu = {:.2f} and sigma = {:.2f}\n".format(mu, sigma))
    plt.title("SalePrice distribution")
    plt.savefig("SalePriceDist.png")
    fig = plt.figure()
    res = stats.probplot(train["SalePrice"], plot = plt)
    plt.savefig("SalePriceProb.png")
    plt.close()
    # 对SalePrice进行对数转换
    train["SalePrice"] = np.log1p(train["SalePrice"])
    # 再画图
    plt.figure()
    sns.distplot(train["SalePrice"], fit = norm)
    (mu, sigma) = norm.fit(train["SalePrice"])
    print("mu = {:.2f} and sigma = {:.2f}\n".format(mu, sigma))
    plt.title("SalePrice distribution")
    plt.savefig("SalePriceDist2.png")
    fig = plt.figure()
    res = stats.probplot(train["SalePrice"], plot = plt)
    plt.savefig("SalePriceProb2.png")
    plt.close()
    # 将训练集和测试集合并到一起
    ntrain = train.shape[0]
    ntest = test.shape[0]
    y_train = train.SalePrice.values
    all_data = pd.concat((train, test)).reset_index(drop = True)
    all_data.drop(["SalePrice"], axis = 1, inplace = True)
    print("all_data的大小为:{}".format(all_data.shape))
    # 处理缺失值
    all_data_na = (all_data.isnull().sum()/len(all_data))*100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending = False)[:30]
    missing_data = pd.DataFrame({"Missing Ratio" : all_data_na})
    print(missing_data.head(20))
    
    # 画图看看
    f, ax = plt.subplots(figsize = (15, 12))
    plt.xticks(rotation = "90")
    sns.barplot(x = all_data_na.index, y = all_data_na)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)
    plt.savefig("Missingdata.png")
    plt.close()
    
    # 特征与SalePrice的相关性
    corrmat = train.corr()
    plt.subplots(figsize=(12,9))
    sns.heatmap(corrmat, vmax = 0.9, square = True)
    plt.savefig("corrmat.png")
    
    # 处理缺失值
    # PoolQC 缺失值代表没游泳池
    all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
    
    # MiscFeature 缺失值代表没该特征
    all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
    
    # Alley 缺失值代表没有小巷入口
    all_data["Alley"] = all_data["Alley"].fillna("None")
    
    # Fence 缺失值代表没栅栏
    all_data["Fence"] = all_data["Fence"].fillna("None")
    
    # FireplaceQu 缺失值代表没壁炉
    all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
    
    # LotFrontage 用其邻居的临街面积的中位数填充缺失值
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x : x.fillna(x.median()))
    
    # GarageType, GarageFinish, GarageQual and GarageCond 都替换为None
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        all_data[col] = all_data[col].fillna("None")
        
    # GarageYrBlt, GarageArea and GarageCars 替换为0
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        all_data[col] = all_data[col].fillna(0)
        
    # BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath 没有地下室，置为0
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        all_data[col] = all_data[col].fillna(0)
    
    # BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 没有地下室，置为None
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        all_data[col] = all_data[col].fillna("None")
        
    # MasVnrArea and MasVnrType 缺失值代表没有砖石覆盖，z置为0和None
    all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
    all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
    
    # MSZoning 用最多的值"RL"代替
    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
    
    # Utilities大多数值为AllPub，只有一个NoSeWa和两个NA，由于NoSeWa只在训练集中出现，可以安全去除。
    all_data = all_data.drop(['Utilities'], axis=1)
    
    # Functional缺失值代表是典型的。
    all_data["Functional"] = all_data["Functional"].fillna("Typ")
    
    # Electrical只有一个缺失值，用众数代替
    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
    
    # KitchenQual只有一个缺失值，用众数代替
    all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
    
    # Exterior1st and Exterior2nd 用众数代替
    all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
    all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
    
    # SaleType 用众数填充
    all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
    
    # MSSubClass 用None填充
    all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
    
    # OK，再看看有没有缺失值的
    all_data_na = (all_data.isnull().sum()/len(all_data))*100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending = False)[:30]
    missing_data = pd.DataFrame({"Missing Ratio" : all_data_na})
    print(missing_data.head())
    
    # 更多的特征工程
    # 转换实际上是分类变量的数值变量
    # MSSubClass
    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
    
    # OverallCond
    all_data['OverallCond'] = all_data['OverallCond'].astype(str)
    
    # 售卖年份和月份
    all_data['YrSold'] = all_data['YrSold'].astype(str)
    all_data['MoSold'] = all_data['MoSold'].astype(str)
    
    # 对一些分类变量进行标签编码
    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(all_data[c].values))
        
        all_data[c] = lbl.transform(list(all_data[c].values))
        
    print('Shape all_data: {}'.format(all_data.shape))
    
    # 将所有面积特征相加
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    
    # 处理偏态特征
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    # 检查所有数值特征的偏态性
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    print("数值特征的偏态性:")
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    print(skewness.head(10))
    
    # 进行Box Cox转换
    skewness = skewness[abs(skewness) > 0.75]
    print("有{}个数值特征要进行Box Cox转换".format(skewness.shape[0]))
    
    from scipy.special import boxcox1p
    skewed_features = skewness.index
    lam = 0.15
    
    for feat in skewed_features:
        all_data[feat] = boxcox1p(all_data[feat], lam)
    all_data = pd.get_dummies(all_data)
    print(all_data.shape)
    
    # 最后，重新划分训练集和测试集
    train = all_data[:ntrain]
    test = all_data[ntrain:]
