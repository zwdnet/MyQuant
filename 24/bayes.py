# coding:utf-8
# 朴素贝叶斯的例子


import pandas as pd
from sklearn.naive_bayes import MultinomialNB


# 数据转换
def transform(data):
    data.loc[data.Age == "青少年", "Age"] = 1
    data.loc[data.Age == "中年", "Age"] = 2
    data.loc[data.Age == "老年", "Age"] = 3
    data.loc[data.Income == "高", "Income"] = 1
    data.loc[data.Income == "中", "Income"] = 2
    data.loc[data.Income == "低", "Income"] = 3
    data.loc[data.Alone == "是", "Alone"] = 1
    data.loc[data.Alone == "否", "Alone"] = 2
    data.loc[data.Credit == "良好", "Credit"] = 1
    data.loc[data.Credit == "一般", "Credit"] = 2
    data.loc[data.Buy == "是", "Buy"] = 1
    data.loc[data.Buy == "否", "Buy"] = 2
    return data


if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    print(data)
    data = transform(data)
    print(data)
    train_data = data[["Age", "Income", "Alone", "Credit"]]
    test_data = data["Buy"]
    
    # test = pd.Series({"Age":3, "Income":3, "Alone":2, "Credit":2})
    test = np.array([3, 3, 2, 2])
    print(test)
    
    print("测试输出")
    print(data.values)
    # 进行朴素贝叶斯分类模型训练
    clf = MultinomialNB(alpha = 2.0)
    clf.fit(data.values, test_data)
    print(clf.class_log_prior_)
    # 用模型预测
    print(clf.predict(test))
    print(clf.predict_proba(test))
    