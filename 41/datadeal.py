# coding:utf-8
# python数据操作实操


import pandas as pd


if __name__ == "__main__":
    # 1.初始化，存储，加载数据
    # 创建DataFrame
    df1 = pd.DataFrame({
    "工资":[5000, 7000, 9000, 8500],
    "绩效分":[60, 84, 98, 91],
    "备注":["不及格", "良好", "最佳", "优秀"],
    "姓名":["老王", "小刘", "小赵", "老龚"]}
    )
    df1.set_index("姓名", inplace = True)
    print(df1)
    # 存储数据
    df1.to_csv("data.csv")
    df1.to_excel("data.xlsx")
    # 读取数据
    df2 = pd.read_csv("data.csv", index_col = "姓名")
    print(df2)
    # 2.认识数据
    print(df2.head(), df2.tail(), df2.sample(2), df2.info())
    # 统计信息概览
    print(df2.describe())
    # 3.列的基本操作方式
    # 增
    df2["新列"] = range(1, len(df2)+1)
    print(df2)
    # 删
    df2.drop("新列", axis = 1, inplace = True)
    print(df2)
    # 选某列
    print(df2["绩效分"])
    # 选多列
    print(df2[["绩效分", "工资"]])
    # 改
    df2["绩效分"] = [10, 20, 40, 30]
    print(df2)
    #4.索引方式
    #基于位置(数字)的索引
    print(df2.iloc[1:3, :])
    print(df2.iloc[1:4, 2:3])
    print(df2.iloc[:, [0, 2]])
    #基于名称(标签)的索引
    print(df2["绩效分"])
    print(df2["绩效分"] >= 30)
    print(df2.loc[df2["绩效分"] >= 30])
    print(df2.loc[:, ["绩效分", "备注"]])
    print(df2.loc[df2["工资"].isin([5000, 7000]), ["绩效分", "备注"]])
    print(df2["工资"].mean())
    print(df2["工资"].std())
    print(df2["工资"].median())
    print(df2["工资"].max())
    print(df2["工资"].min())
    print(df2.loc[(df2["工资"] > df2["工资"].mean()) & (df2["绩效分"] > 30)])
    #5.数据清洗
    #导入数据
    d1 = pd.read_excel("清洗数据集.xlsx", sheet_name = "一级流量")
    d2 = pd.read_excel("清洗数据集.xlsx", sheet_name = "二级流量")
    d3 = pd.read_excel("清洗数据集.xlsx", sheet_name = "三级流量")
    print(d1.head(2), d2.head(2), d3.head(2))
    # 增 拓展数据维度
    # 纵向合并
    df = pd.concat([d1, d2, d3])
    print(df)
    # 横向合并
    h1 = pd.DataFrame({"语文":[1,3,5,7,9],
     "数学":[2,4,6,8,10], "英语":[5,4,3,2,1]}, index = ["a", "b", "c", "d", "e"])
    h2 = pd.DataFrame({"物理":[1,6,5,1],
     "化学":[2,3,2,10]}, index = ["c", "d", "e", "f"])
    h = pd.merge(left = h1, right = h2, left_index = True, right_index = True, how = "inner")
    print(h)
    h = pd.merge(left = h1, right = h2, left_index = True, right_index = True, how = "outer")
    print(h)
    # 删 删空去重
    # 删空
    print(df.dropna())
    # 去重
    repeat = pd.concat([df, df])
    print(len(repeat))
    print(df.drop_duplicates(subset = "流量级别"))
    # 指定要保留的行
    print(df.drop_duplicates(subset = "流量级别", keep = "last"))
    # 查，基于条件查询
    # 按条件索引/筛选
    print(df.loc[(df["访客数"] > 10000) & (df["流量级别"] == "一级"), :])
    # 排序
    sort_df = df.sort_values("支付金额", ascending = False)
    print(sort_df)
    # 最好将改变后的数据赋给新变量，而不是设置inplace参数
    # 分
    # 分组
    print(df.groupby("流量级别").sum())
    # 不把分组列设为索引列
    print(df.groupby("流量级别", as_index = False)[["访客数", "支付金额"]].sum())
    # 切分
    df["分类打标"] = pd.cut(x = df["访客数"], right = False, bins = [0, 100, 1000, 10000, 100000], labels = ["辣鸡", "百级", "千级", "万级"])
    print(df)
    # 6.apply函数
    score = pd.read_excel("apply案例数据.xlsx", sheet_name = "成绩表")
    print(score.head(6))
    max_score = score.groupby("姓名")["综合成绩"].apply(max).reset_index()
    print(max_score)
    min_score = score.groupby("姓名")["综合成绩"].apply(min).reset_index()
    print(min_score)
    # 两张表按姓名合并
    score_combine = pd.merge(max_score, min_score, left_on = "姓名", right_on = "姓名", how = "inner")
    print(score_combine)
    # 另一个例子，求第三
    order = pd.read_excel("apply案例数据.xlsx", sheet_name = "省市销售数据")
    print(order.head())
    print(order.info())
    order_rank = order.sort_values(["省份", "近1月销售额"], ascending = False)
    print(order_rank)
    def get_third(x):
        if len(x) <= 1:
            return x.iloc[0, :]
        else:
            return x.iloc[2, :]
    result = order_rank.groupby("省份")[["城市", "近1月销售额"]].apply(get_third)
    print(result)
