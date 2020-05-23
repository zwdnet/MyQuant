# coding:utf-8
# 《programming for the puzzled》实操
# 8.晚宴的人数


# 生成宾客列表的组合
def Combinations(n, guestList):
    allCombL = []
    for i in range(2**n):
        num = i
        cList = []
        for j in range(n):
            if num%2 == 1:
                cList = [guestList[n-1-j]] + cList
            num = num // 2
        allCombL.append(cList)
    return allCombL
    
    
# 排除不符合的组合
def removeBadCombs(allCombL, dislikePairs):
    allGoodCombs = []
    for i in allCombL:
        good = True
        for j in dislikePairs:
            if j[0] in i and j[1] in i:
                good = False
        if good:
            allGoodCombs.append(i)
    return allGoodCombs
    
    
# 找到元素最多的组合，就是邀请名单啦
def InviteDinner(guestList, dislikePairs):
    allCombL = Combinations(len(guestList), guestList)
    allGoodCombs = removeBadCombs(allCombL, dislikePairs)
    invite = []
    invite = max(allGoodCombs, key = len)
    print("邀请名单:", invite)
    
    
# 优化内存占用
def InviteDinnerOptimized(guestList, dislikePairs):
    n, invite = len(guestList), []
    for i in range(2**n):
        Combination = []
        num = i
        for j in range(n):
            if (num%2 == 1):
                Combination = [guestList[n-1-j]] + Combination
            num = num // 2
        good = True
        for j in dislikePairs:
            if j[0] in Combination and j[1] in Combination:
                good = False
        if good:
            if len(Combination) > len(invite):
                invite = Combination
    print("邀请名单:", invite)
    
    
# 练习1，带权重的算法
def InviteDinnerWeight(guestList, dislikePairs):
    allCombL = Combinations2(len(guestList), guestList)
    allGoodCombinations = removeBadCombinations2(allCombL, dislikePairs)

    invite = max(allGoodCombinations, key=weight)
    print ('带权重的解法:', invite)
    print ('权重为:', weight(invite))
    
    
def Combinations2(n, guestList):
    allCombL = []
    for i in range(2**n):
        num = i
        cList = []
        for j in range(n): 
            if (num % 2 == 1):
                cList = [guestList[n-1-j]] + cList
            num = num // 2
        allCombL.append(cList)
    return allCombL
    
    
def removeBadCombinations2(allCombL, dislikePairs):
    allGoodCombinations = []
    for i in allCombL:
        good = True
        for j in dislikePairs:
            if member(j[0], i) and member(j[1], i):
                good = False
        if good:
            allGoodCombinations.append(i)          
    return allGoodCombinations
    
    
def member(guest, gtuples):
    for g in gtuples:
        if guest == g[0]:
            return True
    return False
    
    
def weight(comb):
    return sum(c[1] for c in comb)


if __name__ == "__main__":
    guest = ["A", "B", "C", "D", "E"]
    combL = Combinations(len(guest), guest)
    print(combL)
    dislikePairs = [["A", "B"],
                              ["B", "C"]]
    goodCombL = removeBadCombs(combL, dislikePairs)
    print(goodCombL)
    
    InviteDinner(guest, dislikePairs)
    
    # 更大规模的问题
    print("更大规模的问题")
    LargeDislikes = [['B', 'C'], ['C', 'D'], ['D', 'E'], ['F', 'G'], ['F', 'H'], ['F', 'I'], ['G', 'H']]
    LargeGuestList = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    InviteDinner(LargeGuestList, LargeDislikes)
    
    InviteDinnerOptimized(LargeGuestList, LargeDislikes)
    # 练习1
    dislikePairs = [['Alice','Bob'],['Bob','Eve']]
    guestList = [('Alice', 2), ('Bob', 6), ('Cleo', 3), ('Don', 10), ('Eve', 3)]
    InviteDinnerWeight(guestList, dislikePairs)
    LargeDislikes = [ ['B', 'C'], ['C', 'D'], ['D', 'E'], ['F', 'G'], ['F', 'H'], ['F', 'I'], ['G', 'H'] ]
    LargeGuestList = [('A', 2), ('B', 1), ('C', 3), ('D', 2), ('E', 1), ('F', 4), ('G', 2), ('H', 1), ('I', 3)]
    InviteDinnerWeight(LargeGuestList, LargeDislikes)

    