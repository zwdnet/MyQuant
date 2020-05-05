# coding:utf-8
# 《programming for the puzzled》实操
# 2.参加聚会的最佳时间


# 自己的方法
def bestTime(sched):
    n = len(sched)
    start = []
    end = []
    for t in sched:
        start.append(t[0])
        end.append(t[1])
    minTime = min(start) # 最早的开始时间
    maxTime = max(end) # 最晚的结束时间
    max_time = 0
    result = minTime
    for time in range(minTime, maxTime):
        times = 0
        for i in range(n):
            if time >= start[i] and time+1 <= end[i]:
                times += 1
        if times > max_time:
            max_time = times
            result = time
        times = 0
    print(max_time)
    return result
    
    
# 作者的第一个算法，也是穷举
def bestTimeToParty(schedule):
    start = schedule[0][0]
    end = schedule[0][1]
    for c in schedule:
        start = min(c[0], start)
        end = max(c[1], end)
        count = celebrityDensity(schedule, start, end)
        maxcount = 0
        for i in range(start, end+1):
            if count[i] > maxcount:
                maxcount = count[i]
                time = i
    print("到达party的最佳时间为", time, "点，有", maxcount, "个人抵达。")
        
        
# 工具函数
def celebrityDensity(sched, start, end):
    count = [0]*(end+1)
    for i in range(start, end+1):
        count[i] = 0
        for c in sched:
            if c[0] <= i and c[1] > i:
                count[i] += 1
    return count
    
    
# 处理更细的时间
def bestTimeToPartySmart(schedule):
    times = []
    for c in schedule:
        times.append((c[0], "start"))
        times.append((c[1], "end"))
    sortList(times)
    maxCount, time = chooseTime(times)
    print("到达聚会的最佳时间为{}点，能遇到{}个人。".format(time, maxCount))
    
    
# 对列表排序
def sortList(tlist):
    for ind in range(len(tlist)-1):
        iSm = ind
        for i in range(ind, len(tlist)):
            if tlist[iSm][0] > tlist[i][0]:
                iSm = i
        tlist[ind], tlist[iSm] = tlist[iSm], tlist[ind]
        
        
# 选择时间
def chooseTime(times):
    rcount = 0
    maxcount = time = 0
    for t in times:
        if t[1] == "start":
            rcount += 1
        elif t[1] == "end":
            rcount -= 1
        if rcount > maxcount:
            maxcount = rcount
            time = t[0]
    return maxcount, time
    
    
# 练习1. 增加时间限制
def bestTimeToPartySmart2(schedule, ystart, yend):
    times = []
    for c in schedule:
        times.append((c[0], "start"))
        times.append((c[1], "end"))
    sortList(times)
    maxCount, time = chooseTime2(times, ystart, yend)
    print("到达聚会的最佳时间为{}点，能遇到{}个人。".format(time, maxCount))
    
    
# 选择时间，增加时间限制
def chooseTime2(times, ystart, yend):
    rcount = 0
    maxcount = time = 0
    for t in times:
        if t[1] == "start":
            rcount += 1
        elif t[1] == "end":
            rcount -= 1
        if rcount > maxcount  and t[0] >= ystart and t[0] < yend:
            maxcount = rcount
            time = t[0]
    return maxcount, time
    
    
# 练习2.另一种不依赖时间粒度的算法
def bestTimeToPartySmart3(schedule):
    maxCount = 0
    time = 0
    n = len(schedule)
    for i in range(n):
        count = 0
        start = schedule[i][0]
        for j in range(n):
            if schedule[j][0] <= start and schedule[j][1] > start:
                count += 1
        if count > maxCount:
            maxCount = count
            time = start
    print("到达聚会的最佳时间为{}点，能遇到{}个人。".format(time, maxCount))
    
    
# 练习3，使自己见到的嘉宾权重最大
def bestTimeToPartySmart4(schedule):
    maxWeight = 0
    time = 0
    n = len(schedule)
    for i in range(n):
        weight = 0
        start = schedule[i][0]
        for j in range(n):
            if schedule[j][0] <= start and schedule[j][1] > start:
                weight += schedule[j][2]
        if weight > maxWeight:
            maxWeight = weight
            time = start
    print("到达聚会的最佳时间为{}点，最大权重值为{}。".format(time, maxWeight))
    
    
if __name__ == "__main__":
    sched = [(6, 8), (6, 12), (6, 7), (7, 8), (7, 10), (8, 9), (8, 10), (9, 12), (9, 10), (10, 11), (10, 12), (11, 12)]
    # 自己的方法
    result = bestTime(sched)
    print("最佳时间:", result)
    # 作者的方法
    bestTimeToParty(sched)
    # 改进，处理更细的时间段
    sched2 = [(6.0, 8.0), (6.5, 12.0), (6.5, 7.0), (7.0, 8.0), (7.5, 10.0), (8.0, 9.0), (8.0, 10.0), (9.0, 12.0), (9.5, 10.0), (10.0, 11.0), (10.0, 12.0), (11.0, 12.0)]
    bestTimeToPartySmart(sched2)
    # 练习1，增加我自己的时间限制
    bestTimeToPartySmart2(sched2, 10.0, 12.0)
    # 练习2，另一种算法
    bestTimeToPartySmart3(sched2)
    # 练习3，使自己见到的嘉宾权重最大
    sched3 = [(6.0, 8.0, 2), (6.5, 12.0, 1), (6.5, 7.0, 2), (7.0, 8.0, 2), (7.5, 10.0, 3), (8.0, 9.0, 2), (8.0, 10.0, 1), (9.0, 12.0, 2), (9.5, 10.0, 4), (10.0, 11.0, 2), (10.0, 12.0, 3), (11.0, 12.0, 7)]
    bestTimeToPartySmart4(sched3)
    