# coding:utf-8
# kaggle题目泰坦尼克号预测
# 特征工程

import tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = "darkgrid")

import string
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import model_selection

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


# 根据姓名提取姓
def extract_surname(data):
    families = []
    
    for i in range(len(data)):
        name = data.iloc[i]
        
        if '(' in name:
            name_no_bracket = name.split('(')[0]
        else:
            name_no_bracket = name
        
        family = name_no_bracket.split('(')[0]
        title = name_no_bracket.split(',')[1].strip().split(' ')[0]
        
        # 将符号用空格代替
        for c in string.punctuation:
            family = family.replace(c, '').strip()
            
        families.append(family)
        
    return families
    
    
# 对Fare进行工程
def FareEng(df_all):
    # 处理Fare特征
    # 将Fare划分为13组，画图
    df_all["Fare"] = pd.qcut(df_all["Fare"], 13)
    # 绘图
    fig, axs = plt.subplots(figsize=(22, 9))
    sns.countplot(x = "Fare", hue = "Survived", data = df_all)
    plt.xlabel('Fare', size=15, labelpad=20)
    plt.ylabel('Passenger Count', size=15, labelpad=20)
    plt.tick_params(axis='x', labelsize=10)
    plt.tick_params(axis='y', labelsize=15)
    
    plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})
    plt.title('Count of Survival in {} Feature'.format('Fare'), size=15, y=1.05)
    
    plt.savefig("FE_fare.png")
    plt.close()
    
    
# 对Age进行特征工程
def AgeEng(df_all):
     # 将Age划分为10组，画图
    df_all["Age"] = pd.qcut(df_all["Age"], 10)
    # 绘图
    fig, axs = plt.subplots(figsize=(22, 9))
    sns.countplot(x = "Age", hue = "Survived", data = df_all)
    plt.xlabel('Age', size=15, labelpad=20)
    plt.ylabel('Passenger Count', size=15, labelpad=20)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    
    plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})
    plt.title('Count of Survival in {} Feature'.format('Age'), size=15, y=1.05)
    
    plt.savefig("FE_age.png")
    plt.close()
    
    
# 创建Family_Size特征
def FamilyEng(df_all):
    # 创建Family_Size特征 画图分析
    df_all["Family_Size"] = df_all["SibSp"] + df_all["Parch"] + 1
    fig, axs = plt.subplots(figsize=(10, 10), ncols=2, nrows=2)
    #plt.subplots_adjust(right = 1.5)
    
    sns.barplot(x=df_all['Family_Size'].value_counts().index, y=df_all['Family_Size'].value_counts().values, ax=axs[0][0])
    sns.countplot(x='Family_Size', hue='Survived', data=df_all, ax=axs[0][1])
    
    axs[0][0].set_title('Family Size Feature Value Counts', size=10, y=1.05)
    axs[0][1].set_title('Survival Counts in Family Size ', size=10, y=1.05)
    
    family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}
    df_all['Family_Size_Grouped'] = df_all['Family_Size'].map(family_map)
    
    sns.barplot(x=df_all['Family_Size_Grouped'].value_counts().index,y=df_all['Family_Size_Grouped'].value_counts().values, ax=axs[1][0])
    sns.countplot(x='Family_Size_Grouped', hue='Survived', data=df_all, ax=axs[1][1])
    
    axs[1][0].set_title('Family Size Feature Value Counts After Grouping', size=10, y=1.05)
    axs[1][1].set_title('Survival Counts in Family Size After Grouping', size=10, y=1.05)
    
    for i in range(2):
        axs[i][1].legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 10})
        for j in range(2):
            axs[i][j].tick_params(axis='x', labelsize=10)
            axs[i][j].tick_params(axis='y', labelsize=10)
            axs[i][j].set_xlabel('')
            axs[i][j].set_ylabel('')
    plt.savefig("FE_family.png")
    plt.close()
    
    return df_all
    
    
# 创建Ticket_Frequence特征
def TicketEng(df_all):
    # Ticket_Frequency特征，绘图
    df_all["Ticket_Frequency"] = df_all.groupby("Ticket")["Ticket"].transform("count")
    fig, axs = plt.subplots(figsize = (12, 9))
    sns.countplot(x = "Ticket_Frequency", hue = "Survived", data = df_all)
    plt.xlabel('Ticket Frequency', size=15, labelpad=20)
    plt.ylabel('Passenger Count', size=15, labelpad=20)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    
    plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})
    plt.title('Count of Survival in {} Feature'.format('Ticket Frequency'), size=15, y=1.05)
    
    plt.savefig("FE_ticket.png")
    plt.close()
    
    return df_all
    
    
# 根据Name属性新建Title和Is_Married属性
def TitleMarriedEng(df_all):
    # 根据姓名前缀生成Title和Is_Married特征并分析。
    df_all["Title"] = df_all["Name"].str.split(', ', expand = True)[1].str.split('.', expand = True)[0]
    df_all["Is_Married"] = 0
    df_all["Is_Married"].loc[df_all["Title"] == "Mrs"] = 1
    
    fig, axs = plt.subplots(nrows=2, figsize=(20, 20))
    sns.barplot(x=df_all['Title'].value_counts().index, y=df_all['Title'].value_counts().values, ax=axs[0])
    
    axs[0].tick_params(axis='x', labelsize=10)
    axs[1].tick_params(axis='x', labelsize=15)
    
    for i in range(2):
        axs[i].tick_params(axis='y', labelsize=15)
        
    axs[0].set_title('Title Feature Value Counts', size=20, y=1.05)
    
    df_all['Title'] = df_all['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
    df_all['Title'] = df_all['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')
    
    sns.barplot(x=df_all['Title'].value_counts().index, y=df_all['Title'].value_counts().values, ax=axs[1])
    axs[1].set_title('Title Feature Value Counts After Grouping', size=20, y=1.05)
    
    plt.savefig("FE_title.png")
    plt.close()
    
    return df_all
    
    
# 进行Family和Ticket的特征工程
def FamilyTicketEng(df_all):
    # 根据Name特征建立Family特征
    df_all["Family"] = extract_surname(df_all["Name"])
    df_train, df_test = tools.divide_df(df_all)
    dfs = [df_train, df_test]
    # 建立同时出现于训练集和测试集的家庭和票种的列表
    non_unique_families = [x for x in df_train["Family"].unique() if x in df_test["Family"].unique()]
    non_unique_tickets = [x for x in df_train["Ticket"].unique() if x in df_test["Ticket"].unique()]
    
    df_family_survival_rate = df_train.groupby("Family")["Survived", "Family", "Family_Size"].median()
    df_ticket_survival_rate = df_train.groupby("Ticket")["Survived", "Ticket", "Ticket_Frequency"].median()
    
    family_rates = {}
    ticket_rates = {}
    
    # 检查同时在训练集和测试集中且成员数大于1的家庭
    for i in range(len(df_family_survival_rate)):
        if df_family_survival_rate.index[i] in non_unique_families and df_family_survival_rate.iloc[i, 1] > 1:
            family_rates[df_family_survival_rate.index[i]] = df_family_survival_rate.iloc[i, 0]
            
    for i in range(len(df_ticket_survival_rate)):
        if df_ticket_survival_rate.index[i] in non_unique_tickets and df_ticket_survival_rate.iloc[i, 1] > 1:
            ticket_rates[df_ticket_survival_rate.index[i]] = df_ticket_survival_rate.iloc[i, 0]
    
    mean_survival_rate = np.mean(df_train["Survived"])
    
    train_family_survival_rate = []
    train_family_survival_rate_NA = []
    test_family_survival_rate = []
    test_family_survival_rate_NA = []
    
    for i in range(len(df_train)):
        if df_train["Family"][i] in family_rates:
            train_family_survival_rate.append(family_rates[df_train["Family"][i]])
            train_family_survival_rate_NA.append(1)
        else:
            train_family_survival_rate.append(mean_survival_rate)
            train_family_survival_rate_NA.append(0)
            
    for i in range(len(df_test)):
        if df_test["Family"].iloc[i] in family_rates:
            test_family_survival_rate.append(family_rates[df_test["Family"].iloc[i]])
            test_family_survival_rate_NA.append(1)
        else:
            test_family_survival_rate.append(mean_survival_rate)
            test_family_survival_rate_NA.append(0)
            
    df_train["Family_Survival_Rate"] = train_family_survival_rate
    df_train["Family_Survival_Rate_NA"] = train_family_survival_rate_NA
    df_test["Family_Survival_Rate"] = test_family_survival_rate
    df_test["Family_Survival_Rate_NA"] = test_family_survival_rate_NA
    
    train_ticket_survival_rate = []
    train_ticket_survival_rate_NA = []
    test_ticket_survival_rate = []
    test_ticket_survival_rate_NA = []
    
    for i in range(len(df_train)):
        if df_train["Ticket"][i] in ticket_rates:
            train_ticket_survival_rate.append(ticket_rates[df_train["Ticket"][i]])
            train_ticket_survival_rate_NA.append(1)
        else:
            train_ticket_survival_rate.append(mean_survival_rate)
            train_ticket_survival_rate_NA.append(0)
            
    for i in range(len(df_test)):
        if df_test["Ticket"].iloc[i] in ticket_rates:
            test_ticket_survival_rate.append(ticket_rates[df_test["Ticket"].iloc[i]])
            test_ticket_survival_rate_NA.append(1)
        else:
            test_ticket_survival_rate.append(mean_survival_rate)
            test_ticket_survival_rate_NA.append(0)
            
    df_train["Ticket_Survival_Rate"] = train_ticket_survival_rate
    df_train["Ticket_Survival_Rate_NA"] = train_ticket_survival_rate_NA
    df_test["Ticket_Survival_Rate"] = test_ticket_survival_rate
    df_test["Ticket_Survival_Rate_NA"] = test_ticket_survival_rate_NA
    
    for df in [df_train, df_test]:
        df["Survival_Rate"] = (df["Ticket_Survival_Rate"] + df["Family_Survival_Rate"]) / 2
        df["Survival_Rate_NA"] = (df["Ticket_Survival_Rate_NA"] + df["Family_Survival_Rate_NA"]) / 2
        
    df_all = tools.concat_df(df_train, df_test)
    return df_all
    
    
# 对非数值变量进行变换
def dataTransform(df_all):
    # 使用LabelEncoder将分类特征转换为数值类型
    non_numeric_features = ["Embarked", "Sex", "Deck", "Title", "Family_Size_Grouped", "Age", "Fare"]
    df_train, df_test = tools.divide_df(df_all)
    for df in [df_train, df_test]:
        for feature in non_numeric_features:
            df[feature] = LabelEncoder().fit_transform(df[feature])
            
    # 使用独热编码处理分类特征
    cat_features = ["Pclass", "Sex", "Deck", "Embarked", "Title", "Family_Size_Grouped"]
    encoded_features = []
    
    for df in [df_train, df_test]:
        for feature in cat_features:
            encoded_feat = OneHotEncoder().fit_transform(df[feature].values.reshape(-1,1)).toarray()
            n = df[feature].nunique()
            cols = ["{}_{}".format(feature, n) for n in range(1, n+1)]
            encoded_df = pd.DataFrame(encoded_feat, columns = cols)
            encoded_df.index = df.index
            encoded_features.append(encoded_df)
            
    df_train = pd.concat([df_train, *encoded_features[:6]], axis = 1)
    df_test = pd.concat([df_test, *encoded_features[6:]], axis = 1)
    
    # 将数据组合，保留有用的特征
    df_all = tools.concat_df(df_train, df_test)
    drop_cols = ['Deck', 'Embarked', 'Family', 'Family_Size', 'Family_Size_Grouped', 'Name', 'Parch', 'Pclass', 'Sex', 'SibSp', 'Ticket', 'Title', 'Ticket_Survival_Rate','Family_Survival_Rate', 'Ticket_Survival_Rate_NA', 'Family_Survival_Rate_NA']
    df_all.drop(columns = drop_cols, inplace = True)
    
    return df_all
    
    
# 找出最重要的几个特征
def get_top_n_features(df_all, top_n_features):
    # 划分数据
    train_df, test_df = tools.divide_df(df_all)
    titanic_train_data_X = train_df.drop(["Survived", "PassengerId"], axis = 1)
    titanic_train_data_Y = train_df["Survived"]
    
    # random forest
    rf_est = RandomForestClassifier(random_state=0)
    rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [20]}
    rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=-1, cv=10, verbose=1)
    rf_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best RF Params:' + str(rf_grid.best_params_))
    print('Top N Features Best RF Score:' + str(rf_grid.best_score_))
    print('Top N Features RF Train Score:' + str(rf_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_rf = pd.DataFrame({'feature':list(titanic_train_data_X), 'importance': rf_grid.best_estimator_.feature_importances_}).sort_values('importance',ascending=False)
    features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']
    print('Sample 10 Features from RF Classifier')
    print(str(features_top_n_rf[:10]))
    
    # AdaBoost
    ada_est =AdaBoostClassifier(random_state=0)
    ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1]}
    ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=-1, cv=10, verbose=1)
    ada_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best Ada Params:' + str(ada_grid.best_params_))
    print('Top N Features Best Ada Score:' + str(ada_grid.best_score_))
    print('Top N Features Ada Train Score:' + str(ada_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_ada = pd.DataFrame({'feature':list(titanic_train_data_X), 'importance': ada_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']
    print('Sample 10 Feature from Ada Classifier:')
    print(str(features_top_n_ada[:10]))

    # ExtraTree
    et_est = ExtraTreesClassifier(random_state=0)
    et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [20]}
    et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=-1, cv=10, verbose=1)
    et_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best ET Params:' + str(et_grid.best_params_))
    print('Top N Features Best ET Score:' + str(et_grid.best_score_))
    print('Top N Features ET Train Score:' + str(et_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_et = pd.DataFrame({'feature':list(titanic_train_data_X), 'importance': et_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_et =  feature_imp_sorted_et.head(top_n_features)['feature']
    print('Sample 10 Features from ET Classifier:')
    print(str(features_top_n_et[:10]))
    
    # GradientBoosting
    gb_est =GradientBoostingClassifier(random_state=0)
    gb_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1], 'max_depth': [20]}
    gb_grid = model_selection.GridSearchCV(gb_est, gb_param_grid, n_jobs=-1, cv=10, verbose=1)
    gb_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best GB Params:' + str(gb_grid.best_params_))
    print('Top N Features Best GB Score:' + str(gb_grid.best_score_))
    print('Top N Features GB Train Score:' + str(gb_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_gb = pd.DataFrame({'feature':list(titanic_train_data_X), 'importance': gb_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_gb = feature_imp_sorted_gb.head(top_n_features)['feature']
    print('Sample 10 Feature from GB Classifier:')
    print(str(features_top_n_gb[:10]))
    
    # DecisionTree
    dt_est = DecisionTreeClassifier(random_state=0)
    dt_param_grid = {'min_samples_split': [2, 4], 'max_depth': [20]}
    dt_grid = model_selection.GridSearchCV(dt_est, dt_param_grid, n_jobs=-1, cv=10, verbose=1)
    dt_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best DT Params:' + str(dt_grid.best_params_))
    print('Top N Features Best DT Score:' + str(dt_grid.best_score_))
    print('Top N Features DT Train Score:' + str(dt_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_dt = pd.DataFrame({'feature':list(titanic_train_data_X), 'importance': dt_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_dt = feature_imp_sorted_dt.head(top_n_features)['feature']
    print('Sample 10 Features from DT Classifier:')
    print(str(features_top_n_dt[:10]))
    
    # merge the three models
    features_top_n = pd.concat([features_top_n_rf, features_top_n_ada, features_top_n_et, features_top_n_gb, features_top_n_dt],ignore_index=True).drop_duplicates()
    
    features_importance = pd.concat([feature_imp_sorted_rf,feature_imp_sorted_ada, feature_imp_sorted_et,feature_imp_sorted_gb, feature_imp_sorted_dt],ignore_index=True)
    
    return features_top_n, features_importance
    
    
# 进行特征工程
def feature_engineer(df_all):
    FareEng(df_all)
    AgeEng(df_all)
    df_all = FamilyEng(df_all)
    df_all = TicketEng(df_all)
    df_all = TitleMarriedEng(df_all)
    df_all = FamilyTicketEng(df_all)
    df_all = dataTransform(df_all)
    feature_top_n, feature_importance = get_top_n_features(df_all, 10)
    return df_all, feature_top_n
    
