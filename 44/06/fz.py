# coding:utf-8
# 《programming for the puzzled》实操
# 6.找假币


# 比较函数
def compare(groupA, groupB):
    if sum(groupA) > sum(groupB):
        result = "left"
    elif sum(groupA) < sum(groupB):
        result = "right"
    else:
        result = "equal"
    return result
    
    
# 将n个硬币划分为三组，假设n为3的倍数
def splitCoins(coinsList):
    length = len(coinsList)
    group1 = coinsList[0:length//3]
    group2 = coinsList[length//3:length//3*2]
    group3 = coinsList[length//2:length]
    return group1, group2, group3
    
    
# 找到有假币那组
def findFakeGroup(group1, group2, group3):
    result1and2 = compare(group1, group2)
    if result1and2 == "left":
        fakeGroup = group1
    elif result1and2 == "right":
        fakeGroup = group2
    elif result1and2 == "equal":
        fakeGroup = group3
    return fakeGroup
    
# 现在进行分治了
def CoinComparison(coinsList):
    counter = 0
    currList = coinsList
    while len(currList) > 1:
        group1, group2, group3 = splitCoins(currList)
        currList = findFakeGroup(group1, group2, group3)
        counter += 1
    fake = currList[0]
    # 练习1
    if fake == coinsList[0] and fake == coinsList[-1]:
        print("没有假币。")
    else:
        print("假币为第", coinsList.index(fake)+1, "个硬币")
    print("比较次数:", counter)


if __name__ == "__main__":
    coinsList = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 10, 10, 10, 10, 10, 10, 10, 10]
    CoinComparison(coinsList)
    coinsList2 = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    CoinComparison(coinsList2)
    