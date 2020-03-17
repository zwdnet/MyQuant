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


# 逻辑回归
def test_logistic(X_train, Y_train):
    LogModel = LogisticRegression()
    LogModel.fit(X_train, Y_train)
    # 对模型评分
    acc_log = round(LogModel.score(X_train, Y_train)*100, 2)
    return (LogModel, acc_log)


# 尝试各种模型
def model_compare(df_all):
    # 划分数据
    train_df, test_df = tools.divide_df(df_all)
    X_train = train_df.drop("Survived", axis = 1)
    Y_train = train_df["Survived"]
    X_test = test_df
    print(X_train.shape, Y_train.shape, X_test.shape)
    
    # 逻辑回归模型
    logModel, acc_log = test_logistic(X_train, Y_train)
    print("逻辑回归结果:{}".format(acc_log))
    
