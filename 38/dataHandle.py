# coding:utf-8
# kaggle题目泰坦尼克号预测
# 数据清洗，处理和探索


import tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = "darkgrid")


# 计算每个等级的乘客在每个舱位的数量
def get_pclass_dist(df):
    deck_counts = {'A': {}, 'B': {}, 'C': {}, 'D': {}, 'E': {}, 'F': {}, 'G': {}, 'M': {}, 'T': {}}
    decks = df.columns.levels[0]
    
    for deck in decks:
        for pclass in range(1, 4):
            try:
                count = df[deck][pclass][0]
                deck_counts[deck][pclass] = count
            except KeyError:
                deck_counts[deck][pclass] = 0
    df_decks = pd.DataFrame(deck_counts)
    deck_percentages = {}
     
    # 计算每个乘客等级在每个客舱的比例
    for col in df_decks.columns:
        deck_percentages[col] = [(count/df_decks[col].sum()) * 100 for count in df_decks[col]]
         
    return deck_counts, deck_percentages
     

# 绘图显示等级舱位距离
def display_pclass_dist(percentages):
    df_percentages = pd.DataFrame(percentages).transpose()
    deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M', 'T')
    bar_count = np.arange(len(deck_names))
    bar_width = 0.85
    
    pclass1 = df_percentages[0]
    pclass2 = df_percentages[1]
    pclass3 = df_percentages[2]
    
    plt.figure(figsize = (20, 10))
    plt.bar(bar_count, pclass1, color = '#b5ffb9', edgecolor = "white",width = bar_width, label = "Class 1")
    plt.bar(bar_count, pclass2, bottom = pclass1, color = "#f9bc86", edgecolor = "white",width = bar_width, label = "Class 2")
    plt.bar(bar_count, pclass3, bottom = pclass1 + pclass2, color = "#a3acff", edgecolor = "white",width = bar_width, label = "Class 3")
    
    plt.xlabel("Deck", size = 15, labelpad = 20)
    plt.xlabel("Passenger Class Percentage", size = 15, labelpad = 20)
    plt.xticks(bar_count, deck_names)
    plt.tick_params(axis = "x", labelsize = 15)
    plt.tick_params(axis = "y", labelsize = 15)
    plt.legend(loc="upper left",bbox_to_anchor=(1, 1), prop={'size': 15})
    plt.title("Passenger Class Distribution in Decks", size=18, y=1.05)
    plt.savefig("pclassdeck.png")
    plt.close()
    
    
# 计算每个船舱的生存比例
def get_survived_dist(df):
    surv_counts = {'A':{}, 'B':{}, 'C':{}, 'D':{}, 'E':{}, 'F':{}, 'G':{}, 'M':{}}
    decks = df.columns.levels[0]
    
    for deck in decks:
        for survive in range(0, 2):
            surv_counts[deck][survive] = df[deck][survive][0]
    
    df_surv = pd.DataFrame(surv_counts)
    surv_percentages = {}
    
    for col in df_surv.columns:
        surv_percentages[col] = [(count / df_surv[col].sum()) * 100 for count in df_surv[col]]
        
    return surv_counts, surv_percentages
    
    
# 绘制每个船舱乘客生存率图
def display_surv_dist(percentages):
    df_survived_percentages = pd.DataFrame(percentages).transpose()
    deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M')
    bar_count = np.arange(len(deck_names))
    bar_width = 0.85
    
    not_survived = df_survived_percentages[0]
    survived = df_survived_percentages[1]
    
    plt.figure(figsize=(20, 10))
    plt.bar(bar_count, not_survived, color='#b5ffb9', edgecolor='white', width=bar_width, label="Not Survived")
    plt.bar(bar_count, survived, bottom=not_survived, color='#f9bc86', edgecolor='white', width=bar_width, label="Survived")
 
    plt.xlabel('Deck', size=15, labelpad=20)
    plt.ylabel('Survival Percentage', size=15, labelpad=20)
    plt.xticks(bar_count, deck_names)    
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 15})
    plt.title('Survival Percentage in Decks', size=18, y=1.05)
    
    plt.savefig("CabinSurvived.png")
    plt.close()


# 填充Age缺失值
def fillAge(df_all):
    # 计算年龄与其它特征的相关性
    df_all_corr = df_all.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
    print(df_all_corr[df_all_corr["Feature 1"] == "Age"])
    
    # 按等级和性别分组计算年龄中位数
    age_by_pclass_sex = df_all.groupby(["Sex", "Pclass"]).median()["Age"]
    
    for pclass in range(1, 4):
        for sex in ["female", "male"]:
            print("分等级乘客年龄中位数为{} {}s: {}".format(pclass, sex, age_by_pclass_sex[sex][pclass]))
    print("所有乘客的年龄中位数为: {}".format(df_all["Age"].median()))
    # 用各组的年龄中位数填充缺失值
    df_all["Age"] = df_all.groupby(["Sex", "Pclass"])["Age"].apply(lambda x : x.fillna(x.median()))
    print(df_all["Age"].isnull().sum())
    return df_all
    
    
# 填充Embarked缺失值
def fillEmbarked(df_all):
    # 查看Embarked缺失值信息
    print(df_all[df_all["Embarked"].isnull()])
    # 根据搜索的真实值用S填充Embarked
    df_all["Embarked"] = df_all["Embarked"].fillna('S')
    return df_all
    
    
# 处理Fare缺失值
def fillFare(df_all):
    # 处理Fare缺失值
    # 输出缺失值情况
    print(df_all[df_all["Fare"].isnull()])
    # 用与其相同等级没有家庭成员的男性的票价的中位数来填充
    med_fare = df_all.groupby(["Pclass", "Parch", "SibSp"]).Fare.median()[3][0][0]
    df_all["Fare"] = df_all["Fare"].fillna(med_fare)
    return df_all
    
    
# 处理Cabin缺失值，转换为新特征
def fillCabin(df_all):
    # 处理Cabin缺失值
    # 乘客的舱位分布图，M代表缺失值。
    df_all["Deck"] = df_all["Cabin"].apply(lambda s : s[0] if pd.notnull(s) else 'M')
    df_all_decks = df_all.groupby(["Deck", "Pclass"]).count().drop(columns = ['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin', 'PassengerId', 'Ticket']).rename(columns = {"Name" : "Count"}).transpose()
    
    # 画图看看每个客舱的乘客等级比例
    all_deck_count, all_deck_per = get_pclass_dist(df_all_decks)
    display_pclass_dist(all_deck_per)
    
    # 将T舱乘客划为A舱
    idx = df_all[df_all["Deck"] == 'T'].index
    df_all.loc[idx, "Deck"] = 'A'
    
    # 计算每个客舱的乘客生存率，绘图
    df_all_decks_survived = df_all.groupby(['Deck', 'Survived']).count().drop(columns=['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass', 'Cabin', 'PassengerId', 'Ticket']).rename(columns={'Name':'Count'}).transpose()
    all_surv_count, all_surv_per = get_survived_dist(df_all_decks_survived)
    display_surv_dist(all_surv_per)
    
    # 将客舱数据按组成比例分组
    df_all["Deck"] = df_all["Deck"].replace(['A', 'B', 'C'], 'ABC')
    df_all["Deck"] = df_all["Deck"].replace(['D', 'E'], 'DE')
    df_all["Deck"] = df_all["Deck"].replace(['F', 'G'], 'FG')
    print(df_all["Deck"].value_counts())
    # 将Cabin丢弃
    df_all.drop(["Cabin"], inplace = True, axis = 1)
    return df_all


# 填充数据缺失值    
def fillna_data(df_all):
    df_all = fillAge(df_all)
    df_all = fillEmbarked(df_all)
    df_all = fillFare(df_all)
    df_all = fillCabin(df_all)
    return df_all
    

# 目标值分布
def target_dist(df_train, df_test):
    # 目标值的分布
    survived = df_train["Survived"].value_counts()[1]
    not_survived = df_train["Survived"].value_counts()[0]
    survived_per = survived / df_train.shape[0] * 100
    not_survived_per = not_survived / df_train.shape[0] * 100
    print('{}名乘客中的{}名获救，占训练集的{:.2f}%。'.format(df_train.shape[0], survived, survived_per))
    print('{}名乘客中的{}名遇难，占训练集的{:.2f}%。'.format(df_train.shape[0], not_survived, not_survived_per))
    
    plt.figure(figsize=(10, 8))
    sns.countplot(df_train["Survived"])
    plt.xlabel("Survival", size = 15, labelpad = 15)
    plt.ylabel('Passenger Count', size=15, labelpad=15)
    plt.xticks((0, 1), ['Not Survived ({0:.2f}%)'.format(not_survived_per), 'Survived ({0:.2f}%)'.format(survived_per)])
    plt.tick_params(axis='x', labelsize=13)
    plt.tick_params(axis='y', labelsize=13)
    
    plt.title("Training Set Survival Distribution")
    plt.savefig("surviveddist.png")
    plt.close()    
    
    
# 分析特征间的相关性
def features_corr(df_train, df_test):
    # 分析特征间的相关性
    df_train_corr = df_train.drop(["PassengerId"], axis = 1).corr().abs().unstack().sort_values(kind = "quicksort", ascending = False).reset_index()
    df_train_corr.rename(columns = {"level_0": "Feature 1",  "level_1": "Feature 2", 0: "Correlation Coefficient"}, inplace = True)
    df_train_corr.drop(df_train_corr.iloc[1::2].index, inplace = True)
    df_train_corr_nd = df_train_corr.drop(df_train_corr[df_train_corr["Correlation Coefficient"] == 1.0].index)
    
    df_test_corr = df_test.drop(["PassengerId"], axis = 1).corr().abs().unstack().sort_values(kind = "quicksort", ascending = False).reset_index()
    df_test_corr.rename(columns = {"level_0": "Feature 1",  "level_1": "Feature 2", 0: "Correlation Coefficient"}, inplace = True)
    df_test_corr.drop(df_test_corr.iloc[1::2].index, inplace = True)
    df_test_corr_nd = df_test_corr.drop(df_test_corr[df_test_corr["Correlation Coefficient"] == 1.0].index)
    
    # 训练集的高相关性
    corr = df_train_corr_nd["Correlation Coefficient"] > 0.1
    print(df_train_corr_nd[corr])
    # 测试集的高相关性
    corr = df_test_corr_nd["Correlation Coefficient"] > 0.1
    print(df_test_corr_nd[corr])
    
    # 绘相关性图
    fig = plt.figure(figsize = (20, 20))
    sns.heatmap(df_train.drop(["PassengerId"], axis = 1).corr(), annot = True, square = True, cmap = "coolwarm", annot_kws = {"size" : 14})
    plt.tick_params(axis = "x", labelsize = 14)
    plt.title("Training Set Correlations", size = 15)
    plt.savefig("TrainFeatureCorr.png")
    
    fig = plt.figure(figsize = (20, 20))
    sns.heatmap(df_test.drop(["PassengerId"], axis = 1).corr(), annot = True, square = True, cmap = "coolwarm", annot_kws = {"size" : 14})
    plt.tick_params(axis = "y", labelsize = 14)
    plt.title("Testing Set Correlations", size = 15)
    plt.savefig("TestFeatureCorr.png")
    plt.close()
    
    
# 连续型特征分布
def con_features_dist(df_train, df_test):
    # 研究连续型特征的分布
    cont_features = ["Age", "Fare"]
    surv = df_train["Survived"] == 1
    fig, axs = plt.subplots(ncols = 1, nrows = 6, figsize = (15, 15))
    plt.subplots_adjust(right = 1.5)
    
    # 特征中的获救人数分布
    sns.distplot(df_train[~surv]["Age"], label = "Not Survived", hist = True, color='#e74c3c', ax=axs[0])
    axs[0].set_title("Age_Survived dist")
    sns.distplot(df_train[surv]["Fare"], label='Survived', hist=True, color='#2ecc71', ax=axs[1])
    axs[1].set_title("Fare_Survived dist")
    # 数据集中的获救人数分布
    sns.distplot(df_train["Age"], label='Training Set', hist=False, color='#e74c3c', ax=axs[2])
    axs[2].set_title("TrainSetAge_Survived dist")
    sns.distplot(df_test["Age"], label='Test Set', hist=False, color='#2ecc71', ax=axs[3])
    axs[3].set_title("TestSetAge_Survived dist")
    
    sns.distplot(df_train["Fare"], label='Training Set', hist=False, color='#e74c3c', ax=axs[4])
    axs[4].set_title("TrainSetFare_Survived dist")
    sns.distplot(df_test["Fare"], label='Test Set', hist=False, color='#2ecc71', ax=axs[5])
    axs[5].set_title("TestSetFare_Survived dist")

    plt.savefig("feature_dist.png")
    plt.close()
    
    
# 分类特征的分布
def cat_features_dist(df_train, df_test):
    # 研究分类特征
    cat_features = ['Embarked', 'Parch', 'Pclass', 'Sex', 'SibSp', 'Deck']
    
    fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(20, 20))
    plt.subplots_adjust(right=1.5, top=1.25)
    
    for i, feature in enumerate(cat_features, 1):
        plt.subplot(2, 3, i)
        sns.countplot(x=feature, hue='Survived', data=df_train)
        
        plt.xlabel('{}'.format(feature), size=20, labelpad=15)
        plt.ylabel('Passenger Count', size=20, labelpad=15)    
        plt.tick_params(axis='x', labelsize=20)
        plt.tick_params(axis='y', labelsize=20)
        
        plt.legend(['Not Survived', 'Survived'], loc='upper center', prop={'size': 18})
        plt.title('Count of Survival in {} Feature'.format(feature), size=20, y=1.05)
        
    plt.savefig("cat_feature_dist.png")
    plt.close()
    
        
# 数据探索
def exploratory_data(df_all):
    # 划分训练集和测试集
    df_train, df_test = tools.divide_df(df_all)
    # dfs = [df_train, df_test]
    # 目标值分布
    target_dist(df_train, df_test)
    # 相关性分析
    features_corr(df_train, df_test)
    # 连续型特征分布
    con_features_dist(df_train, df_test)
    # 分类特征的分布
    cat_features_dist(df_train, df_test)
    