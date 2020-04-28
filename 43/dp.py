# coding:utf-8
# 动态规划

import numpy as np
import pandas as pd

# 物品类
class Dongxi:
    def __init__(self, name, weight, value):
        self.name = name
        self.weight = weight
        self.value = value
        
    def __str__(self):
        return self.name
        
        
# 背包类
class Bag:
    def __init__(self, volume):
        self.volume = volume
        
        
# 初始化，建立Dongxi类
def Initial(name, weight, value):
    list_of_thing = []
    for i in range(len(name)):
        list_of_thing.append(Dongxi(name[i], weight[i], value[i]))
    return list_of_thing
    
    
# 计算一个list里的对象的总价值
def value_sum(a_list):
    if type(a_list) != list:
        a_list = [a_list]
    return sum([x.value for x in a_list])
    

# 实施动态规划
def begin(list1, bag1, None_type):
    mat1 = pd.DataFrame(np.array([None_type]*(len(list1)+1)*(bag1.volume+1)).reshape(len(list1)+1, bag1.volume+1))
    for i in range(mat1.shape[0]):
        mat1.loc[i, :] = mat1.loc[i,:].apply(lambda x:list(set([x])))
        
    for i in range(1, mat1.shape[0]):
        for j in range(1, mat1.shape[1]):
            if  j < list1[i-1].weight:
                mat1.loc[i, j] = mat1.loc[i-1, j].copy()
            else:
                if value_sum(mat1.loc[i-1,j]) >= value_sum(mat1.loc[i-1, j-list1[i-1].weight]) + list1[i-1].value:
                    mat1.loc[i,j] = mat1.loc[i-1,j].copy()
                else:
                    mat1.loc[i,j] = mat1.loc[i-1, j-list1[i-1].weight].copy()
                    mat1.loc[i,j].append(list1[i-1])
                    if None_type in mat1.loc[i,j] and len(mat1.loc[i,j]) > 1:
                        mat1.loc[i,j].remove(None_type)
    return mat1
    
    
if __name__ == "__main__":
    # 动态规划解决背包问题
    name = ['a','b','c','d','e']
    weight = [2,2,6,5,4]
    value = [6,3,5,4,6]
    None_type = Dongxi('None_t',0,0)
    list1 = Initial(name,weight,value)
    bag1 = Bag(10)
    s1 = begin(list1,bag1,None_type)
    print(s1)