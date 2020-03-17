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


# 尝试各种模型
def model_compare(df_all):
    # 划分数据
    train_df, test_df = tools.divide_df(df_all)
    X_train = train_df.drop("Survived", axis = 1)
    Y_train = train_df["Survived"]
    X_test = test_df
    print(X_train.shape, Y_train.shape, X_test.shape)
    print(X_train.info())
    print(X_test.info())
