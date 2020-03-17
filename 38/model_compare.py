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


# 模型测试
def ModelTest(Model, X_train, Y_train):
    Model.fit(X_train, Y_train)
    # 对模型评分
    acc_result = round(Model.score(X_train, Y_train)*100, 2)
    return acc_result


# 尝试各种模型
def model_compare(df_all):
    # 划分数据
    train_df, test_df = tools.divide_df(df_all)
    X_train = train_df.drop("Survived", axis = 1)
    Y_train = train_df["Survived"]
    X_test = test_df
    print(X_train.shape, Y_train.shape, X_test.shape)
    
    # 逻辑回归模型
    LogModel = LogisticRegression()
    acc_log = ModelTest(LogModel, X_train, Y_train)
    print("逻辑回归结果:{}".format(acc_log))
    
    # SVM支持向量机模型
    SVMModel = SVC()
    acc_svc = ModelTest(SVMModel, X_train, Y_train)
    print("支持向量机结果:{}".format(acc_svc))
    
    # knn算法
    knnModel = KNeighborsClassifier(n_neighbors = 3)
    acc_knn = ModelTest(knnModel, X_train, Y_train)
    print("knn结果:{}".format(acc_knn))
    
    # 朴素贝叶斯模型
    BYSModel = GaussianNB()
    acc_bys = ModelTest(BYSModel, X_train, Y_train)
    print("朴素贝叶斯算法结果:{}".format(acc_bys))
    
    # 感知机算法
    percModel = Perceptron()
    acc_perc = ModelTest(percModel, X_train, Y_train)
    print("感知机算法算法结果:{}".format(acc_perc))
    
    # 线性分类支持向量机
    lin_svcModel = LinearSVC()
    acc_lin_svc = ModelTest(lin_svcModel, X_train, Y_train)
    print("线性分类支持向量机算法结果:{}".format(acc_lin_svc))
    
    # 梯度下降分类算法
    sgdModel = SGDClassifier()
    acc_sgd = ModelTest(sgdModel, X_train, Y_train)
    print("梯度下降分类算法结果:{}".format(acc_sgd))
    
    # 决策树算法
    treeModel = DecisionTreeClassifier()
    acc_tree = ModelTest(treeModel, X_train, Y_train)
    print("决策树算法结果:{}".format(acc_tree))
    
    # 随机森林算法
    forestModel = RandomForestClassifier()
    acc_rand = ModelTest(forestModel, X_train, Y_train)
    print("随机森林算法结果:{}".format(acc_rand))
    
    # 模型评分
    models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'Perceptron', 'Stochastic Gradient Decent', 'Linear SVC','Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, acc_rand, acc_bys, acc_perc, acc_sgd, acc_lin_svc, acc_tree]})
    print(models.sort_values(by='Score', ascending=False))
    
    # 用决策树模型进行预测
    tools.Submission(treeModel, test_df, "decisetree.csv")
    
