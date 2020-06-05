# coding:utf-8
# 《programming for the puzzled》实操
# 13.快速排序


def quicksort(lst, start, end):
    if start < end:
        split = pivotPartitionClever(lst, start, end)
        quicksort(lst, start, split-1)
        quicksort(lst, split+1, end)
        
        
# 划分枢纽的过程
def pivotPartition(lst, start, end):
    pivot = lst[end]
    less, pivotList, more = [], [], []
    for e in lst:
        if e < pivot:
            less.append(e)
        elif e > pivot:
            more.append(e)
        else:
            pivotList.append(e)
    i = 0
    for e in less:
        lst[i] = e
        i += 1
    for e in pivotList:
        lst[i] = e
        i += 1
    for e in more:
        lst[i] = e
        i += 1
    return lst.index(pivot)
    
    
# 计数器
count = 0
# 更好的选取枢纽的方法
def pivotPartitionClever(lst, start, end):
    global count
    pivot = lst[end]
    bottom = start - 1
    top = end
    done = False
    while not done:
        while not done:
            bottom += 1
            if bottom == top:
                done = True
                break
            if lst[bottom] > pivot:
                lst[top] = lst[bottom]
                count += 1
                break
        while not done:
            top -= 1
            if top == bottom:
                done = True
                break
            if lst[top] < pivot:
                lst[bottom] = lst[top]
                count += 1
                break
    lst[top] = pivot
    return top


if __name__ == "__main__":
    a = [4, 65, 2, -31, 0, 99, 83, 782, 1]
    quicksort(a, 0, len(a) - 1)
    print(a)
    print("比较次数:", count)
    