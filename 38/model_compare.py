# coding:utf-8
# 模型比较


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



import tools
import modelevaluation as me
import numpy as np
import pandas as pd



# 尝试各种模型
def model_compare(df_all):
    # 划分数据
    train_df, test_df = tools.divide_df(df_all)
    X_train = train_df.drop(["Survived", "PassengerId"], axis = 1)
    Y_train = train_df["Survived"]
    X_test = test_df
    print(X_train.shape, Y_train.shape, X_test.shape)
    
    # 逻辑回归模型
    LogModel = LogisticRegression()
    acc_log = me.ModelTest(LogModel, X_train, Y_train)
    print("逻辑回归结果:{}".format(acc_log))
    me.learnning_curve(LogModel, "Logistic", df_all, "LearningCurveLogistic.png")
    cross_log = me.cross_val(LogModel, X_train, Y_train, cv = 5)
    print("交叉验证得分:%.3f" % (cross_log))
    
    # SVM支持向量机模型
    SVMModel = SVC()
    acc_svc = me.ModelTest(SVMModel, X_train, Y_train)
    print("支持向量机结果:{}".format(acc_svc))
    me.learnning_curve(SVMModel, "SVC", df_all, "LearningCurveSVC.png")
    cross_svc = me.cross_val(SVMModel, X_train, Y_train, cv = 5)
    print("交叉验证得分:%.3f" % (cross_svc))
    
    # knn算法
    knnModel = KNeighborsClassifier(n_neighbors = 3)
    acc_knn = me.ModelTest(knnModel, X_train, Y_train)
    print("knn结果:{}".format(acc_knn))
    me.learnning_curve(knnModel, "KNN", df_all, "LearningCurveKNN.png")
    cross_knn = me.cross_val(knnModel, X_train, Y_train, cv = 5)
    print("交叉验证得分:%.3f" % (cross_knn))
    
    # 朴素贝叶斯模型
    BYSModel = GaussianNB()
    acc_bys = me.ModelTest(BYSModel, X_train, Y_train)
    print("朴素贝叶斯算法结果:{}".format(acc_bys))
    me.learnning_curve(BYSModel, "Bayes", df_all, "LearningCurveBayes.png")
    cross_bys = me.cross_val(BYSModel, X_train, Y_train, cv = 5)
    print("交叉验证得分:%.3f" % (cross_bys))
    
    # 感知机算法
    percModel = Perceptron()
    acc_perc = me.ModelTest(percModel, X_train, Y_train)
    print("感知机算法算法结果:{}".format(acc_perc))
    me.learnning_curve(percModel, "Perceptron", df_all, "LearningCurvePerceptron.png")
    cross_perc = me.cross_val(percModel, X_train, Y_train, cv = 5)
    print("交叉验证得分:%.3f" % (cross_perc))
    
    # 线性分类支持向量机
    lin_svcModel = LinearSVC()
    acc_lin_svc = me.ModelTest(lin_svcModel, X_train, Y_train)
    print("线性分类支持向量机算法结果:{}".format(acc_lin_svc))
    me.learnning_curve(lin_svcModel, "LinearSVC", df_all, "LearningCurveLinearSVC.png")
    cross_lin_svc = me.cross_val(lin_svcModel, X_train, Y_train, cv = 5)
    print("交叉验证得分:%.3f" % (cross_lin_svc))
    
    # 梯度下降分类算法
    sgdModel = SGDClassifier()
    acc_sgd = me.ModelTest(sgdModel, X_train, Y_train)
    print("梯度下降分类算法结果:{}".format(acc_sgd))
    me.learnning_curve(sgdModel, "SGDC", df_all, "LearningCurveSGDC.png")
    cross_sgd = me.cross_val(sgdModel, X_train, Y_train, cv = 5)
    print("交叉验证得分:%.3f" % (cross_sgd))
    
    # 决策树算法
    treeModel = DecisionTreeClassifier()
    acc_tree = me.ModelTest(treeModel, X_train, Y_train)
    print("决策树算法结果:{}".format(acc_tree))
    me.learnning_curve(treeModel, "DecisionTree", df_all, "LearningCurveDecisionTree.png")
    cross_tree = me.cross_val(treeModel, X_train, Y_train, cv = 5)
    print("交叉验证得分:%.3f" % (cross_tree))
    
    # 随机森林算法
    forestModel = RandomForestClassifier()
    acc_rand = me.ModelTest(forestModel, X_train, Y_train)
    print("随机森林算法结果:{}".format(acc_rand))
    me.learnning_curve(forestModel, "RandomForest", df_all, "LearningCurveRandomForest.png")
    cross_rand = me.cross_val(forestModel, X_train, Y_train, cv = 5)
    print("交叉验证得分:%.3f" % (cross_rand))
    
    # 模型评分
    models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'Perceptron', 'Stochastic Gradient Decent', 'Linear SVC','Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, acc_rand, acc_bys, acc_perc, acc_sgd, acc_lin_svc, acc_tree]})
    print(models.sort_values(by='Score', ascending=False))
    
    # 模型交叉验证评分
    models_cross_val = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'Perceptron', 'Stochastic Gradient Decent', 'Linear SVC','Decision Tree'],
    'Score': [cross_svc, cross_knn, cross_log, cross_rand, cross_bys, cross_perc, cross_sgd, cross_lin_svc, cross_tree]})
    print(models_cross_val.sort_values(by='Score', ascending=False))
    
    # 用决策树模型进行预测
    tools.Submission(treeModel, test_df, "decisetree.csv")
    # 用逻辑回归模型进行预测
    tools.Submission(LogModel, test_df, "Logist.csv")
    
