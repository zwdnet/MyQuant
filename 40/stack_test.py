# coding:utf-8
# Stacking实操
# 扔飞镖问题，预测飞镖是谁扔的。


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression    #逻辑回归
from sklearn.svm import SVC, LinearSVC                 #支持向量机
from sklearn.ensemble import RandomForestClassifier    #随机森林
from sklearn.neighbors import KNeighborsClassifier     #K最邻近算法
from sklearn.naive_bayes import GaussianNB             #朴素贝叶斯
from sklearn.linear_model import Perceptron            #感知机算法             
from sklearn.linear_model import SGDClassifier         #梯度下降分类
from sklearn.tree import DecisionTreeClassifier        #决策树算法
from sklearn.model_selection import StratifiedKFold    #K折交叉切分
from sklearn.model_selection import GridSearchCV       #网格搜索

from sklearn.model_selection import cross_val_score, KFold, train_test_split
from vecstack import stacking
from sklearn.metrics import accuracy_score


# 对模型进行交叉验证
def cross_val(model, X, Y, cv=5):
    scores = cross_val_score(model, X, Y, cv=cv)
    score = scores.mean()
    return score
    
    
# 模型评分
def ModelTest(Model, X_train, Y_train):
    Model.fit(X_train, Y_train)
    # 对模型评分
    acc_result = round(Model.score(X_train, Y_train)*100, 2)
    return acc_result
    
# 用测试集检验模型预测的正确率
def testModel(Model, X_train, Y_train,  X_test, Y_test):
    Model.fit(X_train, Y_train)
    # 用模型对测试集数据进行预测
    res = Model.predict(X_test)
    print(res, Y_test.values)
    n = 0.0
    Y_test_value = Y_test.values
    for i in range(len(Y_test_value)):
        # print(i, res[i], Y_test.iloc[i:1])
        if res[i] == Y_test_value[i]:
            n += 1.0
    score = n/len(Y_test)
    return score
    
    
# 找到一个字符串数组中的众数
#def mostCommon(data):
#    count = {b'M':0, b'S':0, b'B':0, b'K':0}
#    for i in data:
#        count[i] +=1
#    return max(count, key = count.get)
#    
#    
# 使用交叉验证的方法得到次级训练集
#def get_stacking(Model, x_train, y_train, x_test, n_folds=10):
#    """
#这个函数是stacking的核心，使用交叉验证的方法得到次级训练集x_train, y_train, x_test 的值应该为numpy里面的数组类型 numpy.ndarray .如果输入为pandas的DataFrame类型则会把报错"""
#    train_num, test_num = x_train.shape[0], x_test.shape[0]
#    second_level_train_set = np.zeros((train_num,), dtype = np.string_)
#    second_level_test_set = np.zeros((test_num,), dtype = np.string_)
#    test_nfolds_sets = np.zeros((test_num, n_folds), dtype = np.string_)
#    kf = KFold(n_splits=n_folds)
#    
#    for i,(train_index, test_index) in enumerate(kf.split(x_train)):
#        # print(x_train)
#        # input("按任意键继续")
#        x_tra, y_tra = x_train[train_index], y_train[train_index]
#        x_tst, y_tst =  x_train[test_index], y_train[test_index]
#        
#        print("stacking")
#        print(x_tra)
#        print(y_tra)
#        Model.fit(x_tra, y_tra)
#        
#        second_level_train_set[test_index] = Model.predict(x_tst)
#        test_nfolds_sets[:,i] = Model.predict(x_test)
#    # print(test_nfolds_sets)   
#    # 因为是字符串，求平均数失败，试试求众数
#    # second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
#    for i in range(n_folds):
#        second_level_test_set[i] = mostCommon(test_nfolds_sets[:, i])
#    return second_level_train_set, second_level_test_set
        

if __name__ == "__main__":
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")
    print(train_data.head())
    print(train_data.info())
    print(test_data.info())
    # 画图看看
    fig = plt.figure()
    colors = ['b','g','r','orange']
    labels = ['Sue', 'Kate', 'Mark', 'Bob']
    for index in range(4):
        x = train_data.loc[train_data["Competitor"] == labels[index]]["XCoord"]
        y = train_data.loc[train_data["Competitor"] == labels[index]]["YCoord"]
        plt.scatter(x, y, c = colors[index], label = labels[index])
    plt.legend(loc = "best")
    plt.savefig("data.png")
    plt.close()
    
    # 基本模型比较
    # 划分数据
    X_train = train_data.drop(["Competitor", "ID"], axis = 1)
    print(X_train.info())
    Y_train = train_data["Competitor"]
    X_test = test_data.drop(["Competitor", "ID"], axis = 1)
    Y_test = test_data["Competitor"]
    # 将Competitor数据转换为整数
#    for item in len(Y_train):
#        for i in range(4):
#            if Y_train[i] == labels[i]:
#                Y_train[i] = i
#    for item in len(Y_test):
#        for i in range(4):
#            if Y_test[i] == labels[i]:
#                Y_test[i] = i
    # 逻辑回归模型
    LogModel = LogisticRegression()
    acc_log = ModelTest(LogModel, X_train, Y_train)
    print("逻辑回归结果:{}".format(acc_log))
    cross_log = cross_val(LogModel, X_train, Y_train, cv = 5)
    print("交叉验证得分:%.3f" % (cross_log))
    
    # SVM支持向量机模型
    SVMModel = SVC()
    acc_svc = ModelTest(SVMModel, X_train, Y_train)
    print("支持向量机结果:{}".format(acc_svc))
    cross_svc = cross_val(SVMModel, X_train, Y_train, cv = 5)
    print("交叉验证得分:%.3f" % (cross_svc))
    
    # knn算法
    knnModel = KNeighborsClassifier(n_neighbors = 4)
    acc_knn = ModelTest(knnModel, X_train, Y_train)
    print("knn结果:{}".format(acc_knn))
    cross_knn = cross_val(knnModel, X_train, Y_train, cv = 5)
    print("交叉验证得分:%.3f" % (cross_knn))
    
    # 朴素贝叶斯模型
    BYSModel = GaussianNB()
    acc_bys = ModelTest(BYSModel, X_train, Y_train)
    print("朴素贝叶斯算法结果:{}".format(acc_bys))
    cross_bys = cross_val(BYSModel, X_train, Y_train, cv = 5)
    print("交叉验证得分:%.3f" % (cross_bys))
    
    # 感知机算法
    percModel = Perceptron()
    acc_perc = ModelTest(percModel, X_train, Y_train)
    print("感知机算法算法结果:{}".format(acc_perc))
    cross_perc = cross_val(percModel, X_train, Y_train, cv = 5)
    print("交叉验证得分:%.3f" % (cross_perc))
    
    # 线性分类支持向量机
    lin_svcModel = LinearSVC()
    acc_lin_svc = ModelTest(lin_svcModel, X_train, Y_train)
    print("线性分类支持向量机算法结果:{}".format(acc_lin_svc))
    cross_lin_svc = cross_val(lin_svcModel, X_train, Y_train, cv = 5)
    print("交叉验证得分:%.3f" % (cross_lin_svc))
    
    # 梯度下降分类算法
    sgdModel = SGDClassifier()
    acc_sgd = ModelTest(sgdModel, X_train, Y_train)
    print("梯度下降分类算法结果:{}".format(acc_sgd))
    cross_sgd = cross_val(sgdModel, X_train, Y_train, cv = 5)
    print("交叉验证得分:%.3f" % (cross_sgd))
    
    # 决策树算法
    treeModel = DecisionTreeClassifier()
    acc_tree = ModelTest(treeModel, X_train, Y_train)
    print("决策树算法结果:{}".format(acc_tree))
    cross_tree = cross_val(treeModel, X_train, Y_train, cv = 5)
    print("交叉验证得分:%.3f" % (cross_tree))
    
    # 随机森林算法
    forestModel = RandomForestClassifier()
    acc_rand = ModelTest(forestModel, X_train, Y_train)
    print("随机森林算法结果:{}".format(acc_rand))
    cross_rand = cross_val(forestModel, X_train, Y_train, cv = 5)
    print("交叉验证得分:%.3f" % (cross_rand))
    
    # 模型评分
    print("模型评分")
    models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'Perceptron', 'Stochastic Gradient Decent', 'Linear SVC','Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, acc_rand, acc_bys, acc_perc, acc_sgd, acc_lin_svc, acc_tree]})
    print(models.sort_values(by='Score', ascending=False))
    
    # 模型交叉验证评分
    print("模型交叉验证评分")
    models_cross_val = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'Perceptron', 'Stochastic Gradient Decent', 'Linear SVC','Decision Tree'],
    'Score': [cross_svc, cross_knn, cross_log, cross_rand, cross_bys, cross_perc, cross_sgd, cross_lin_svc, cross_tree]})
    print(models_cross_val.sort_values(by='Score', ascending=False))
    
    # 进行stacking
    # 用评分前五名的决策树，随机森林，KNN，朴素贝叶斯，支持向量机算法进行stacking。
    models = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier(n_neighbors = 4),
    GaussianNB(),
    SVC()
    ]
    Y_train.replace({"Sue":1.0, "Mark":2.0, "Kate":3.0, "Bob":4.0}, inplace = True)
    Y_test.replace({"Sue":1.0, "Mark":2.0, "Kate":3.0, "Bob":4.0}, inplace = True)
    x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)
    S_train, S_test = stacking(models, x_train, y_train, x_test, regression=False, mode='oof_pred_bag', needs_proba=False, save_dir=None, metric=accuracy_score, n_folds=4, stratified=True, shuffle=True, random_state=0, verbose=2)
    # 第二级，用逻辑回归
    model = DecisionTreeClassifier()
    model = model.fit(S_train, y_train)
    y_pred = model.predict(S_test)
    print('Final prediction score: [%.8f]' % accuracy_score(y_test, y_pred))
        
    #分别测试
    DT_score = testModel(treeModel, X_train, Y_train,  X_test, Y_test)
    RF_score = testModel(forestModel, X_train, Y_train,  X_test, Y_test)
    KNN_score = testModel(knnModel, X_train, Y_train,  X_test, Y_test)
    NB_score = testModel(BYSModel, X_train, Y_train,  X_test, Y_test)
    SVM_score = testModel(SVMModel, X_train, Y_train,  X_test, Y_test)
    stacking_score = testModel(model, S_train, y_train,  S_test, y_test)
    print("Stacking结果")
    stacking_results = pd.DataFrame({
    '模型': ["决策树", "随机森林","KNN","朴素贝叶斯", "支持向量机", "Stacking"],
    '预测正确率': [DT_score, RF_score, KNN_score, NB_score, SVM_score, stacking_score]})
    print(stacking_results.sort_values(by='预测正确率', ascending=False))
