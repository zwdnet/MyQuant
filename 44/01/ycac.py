# coding:utf-8
# 《programming for the puzzled》实操
# 01.you will all conform


# 获取最少的命令数
def minCommand(cap):
    n = len(cap)
    # 先把"F"变成"B"
    minTimesB = 0
    bF = False
    for i in range(n):
        if cap[i] == "F":
            bF = True
            if i == n-1:
                minTimesB += 1
            continue
        if bF == True:
            minTimesB += 1
        bF = False
    # 再把"B"变成"F"
    minTimesF = 0
    bB = False
    for i in range(n):
        if cap[i] == "B":
            bB = True
            if i == n-1:
                minTimesF += 1
            continue
        if bB == True:
            minTimesF += 1
        bB = False
    return min(minTimesB, minTimesF)
    
    
# 书中第一个解法，先找出相应区间
def pleaseConform(caps):
    start = forward = backward = 0
    intervals = []
    n = len(caps)
    minNum = n
    for i in range(1, n):
        # 帽子方向变了，区间改变
        if caps[start] != caps[i]:
            intervals.append((start, i-1, caps[start]))
            if caps[start] == "F":
                forward += 1
            else:
                backward += 1
            start = i
    # 处理最后一个区间
    intervals.append((start, len(caps)-1, caps[start]))
    if caps[start] == "F":
        forward += 1
    else:
        backward += 1
    if forward < backward:
        flip = "F"
        minNum = forward
    else:
        flip = "B"
        minNum = backward
    for t in intervals:
        if t[2] == flip:
            print("在位置{}到{}的人，翻转你的帽子".format(t[0], t[1]))
    return minNum
    
    
# 书中第二个解法，对解法一的优化
def pleaseConform2(caps):
    start = forward = backward = 0
    intervals = []
    # 在数据末尾增加末尾标志
    caps = caps + ["END"]
    n = len(caps)
    minNum = n
    for i in range(1, n):
        # 帽子方向变了，区间改变
        if caps[start] != caps[i]:
            intervals.append((start, i-1, caps[start]))
            if caps[start] == "F":
                forward += 1
            else:
                backward += 1
            start = i
    if forward < backward:
        flip = "F"
        minNum = forward
    else:
        flip = "B"
        minNum = backward
    for t in intervals:
        if t[2] == flip:
            if t[0] != t[1]:
                print("在位置{}到{}的人，翻转你的帽子".format(t[0], t[1]))
            else:
                print("在位置{}的人，翻转你的帽子".format(t[0], t[1]))
    return minNum
    

# 一遍遍历的程序
def pleaseConformOnepass(caps):
    caps = caps + [caps[0]]
    minNum = 0
    for i in range(1, len(caps)):
        if caps[i] != caps[i-1]:
            if caps[i] != caps[0]:
                print("位置在", i)
            else:
                print("到", i-1, "的人，翻转你的帽子!")
                minNum += 1
    return minNum
    
    
# 一遍遍历的程序,根据练习2修改版
def pleaseConformOnepass2(caps):
    if len(caps) == 0:
        return 0
    caps = caps + [caps[0]]
    minNum = 0
    start = 0
    for i in range(1, len(caps)):
        if caps[i] != caps[i-1]:
            if caps[i] != caps[0]:
                start = i
            else:
                if start != i-1:
                    print("位置在", start, "到", i-1, "的人，翻转你的帽子!")
                else:
                    print("位置在", start,  "的人，翻转你的帽子!")
                minNum += 1
    return minNum
    
    
# 练习三，增加"H"帽子的人，忽略
def pleaseConform3(caps):
    start = forward = backward = 0
    intervals = []
    # 在数据末尾增加末尾标志
    caps = caps + ["END"]
    n = len(caps)
    minNum = n
    for i in range(1, n):
        # 帽子方向变了，区间改变
        if caps[start] != caps[i]:
            intervals.append((start, i-1, caps[start]))
            if caps[i-1] == "F":
                forward += 1
            elif caps[i-1] == "B":
                backward += 1
            start = i
    if forward < backward:
        flip = "F"
        minNum = forward
    else:
        flip = "B"
        minNum = backward
    for t in intervals:
        if t[2] == flip:
            if t[0] != t[1]:
                print("在位置{}到{}的人，翻转你的帽子".format(t[0], t[1]))
            else:
                print("在位置{}的人，翻转你的帽子".format(t[0], t[1]))
    return minNum
    
    
# 练习4 压缩和解压程序
# 压缩
def compress(words):
    str_list = list(words)
    str_list.append(str_list[0])
    start = 0
    code = []
    for i in range(1, len(str_list)):
        if str_list[start] != str_list[i]:
            code.append((start, i-1, str_list[start]))
            start = i
    result = []
    for t in code:
        a = t[0]
        b = t[1]
        w = t[2]
        length = b-a+1
        result.append(str(length))
        result.append(w)
    return "".join(result)
    
    
# 解压缩
def decompress(code):
    result = []
    num = []
    char = ""
    for i in range(len(code)):
        if code[i].isalpha() == False:
            num.append(code[i])
        else:
            nums = int("".join(num))
            num = []
            result.append(code[i]*nums)
    return "".join(result)
        

if __name__ == "__main__":
    # 穷举法
    cap = ["F", "F", "B", "B", "B", "F", "B", "B", "B", "F", "F", "B", "F"]
    result = minCommand(cap)
    print(result)
    # 书中第一个方法
    result = pleaseConform(cap)
    print(result)
    # 对第一个方法的优化
    result = pleaseConform2(cap)
    print(result)
    # 一遍遍历的方法
    result = pleaseConformOnepass(cap)
    print(result)
    # 修改后的一遍遍历
    result = pleaseConformOnepass2(cap)
    print(result)
    result = pleaseConformOnepass2([])
    print(result)
    # 增加戴"H"帽子的人
    print("练习3")
    cap2 = ["F", "F", "B", "H", "B", "F", "B", "B", "B", "F", "H", "F", "F"]
    result = pleaseConform3(cap2)
    print(result)
    # 练习四
    s = "BWWWWWBWWWWWWWWWWWAAKKGGP"
    code = compress(s)
    print("原字符串为:", s)
    print("压缩后的字符串为:", code)
    ds = decompress(code)
    print("解压后的字符串为:", ds)
    
    