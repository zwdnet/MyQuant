# coding:utf-8
# 《programming for the puzzled》实操
# 7.找平方根


# 线性复杂度算法
def findSquareRoot(n):
    if n < 0:
        print("要输入非负整数")
        return -1
    i = 0
    while i*i < n:
        i += 1
    if i*i == n:
        return i
    else:
        print(n, "不是完全平方数")
        return -1


# 改进，增加答案精度，指定精度和步长
def findSquareRoot2(n, eps, step):
    if n < 0:
        print("要输入非负整数")
        return -1, 0
    numGuesses = 0.0
    ans = 0.0
    while n - ans**2 > eps:
        ans += step
        numGuesses += 1
    if abs(n - ans**2) > eps:
        # print("求解", n, "的平方根失败")
        print(n, ans**2, n - ans**2, eps)
        return -1, numGuesses
    else:
        print("b")
        # print(ans, "是", n, "的近似平方根")
        return ans, numGuesses
        
        
# 二分搜索
def bisectionSearchForSquareRoot(n, eps):
    if n < 0:
        print("要输入非负整数")
        return -1, 0
    numGuesses = 0
    low = 0.0
    high = n
    ans = (high + low)/2.0
    while abs(ans**2 - n) >= eps:
        if ans**2 < n:
            low = ans
        else:
            high = ans
        ans = (high + low)/2.0
        numGuesses += 1
    return ans, numGuesses
    
    
# 线性查找
NOTFOUND = -1
Ls = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
def Lsearch(L, value):
    for i in range(len(L)):
        if L[i] == value:
            return i
    return NOTFOUND
    
    
# 二分查找
def bsearch(L, value):
    lo, hi = 0, len(L) - 1
    length  = hi
    while lo <= hi:
        mid = (lo+hi)//2
        if L[mid] < value:
            lo = mid + 1
        elif value < L[mid]:
            hi = mid - 1
        else:
            return mid
        # 练习2
        length = hi-lo
        print("当前搜索区间长度:", length)
    return NOTFOUND
    
    
# 练习1.二分搜索改进版
def bisectionSearchForSquareRoot2(n, eps):
    if n < 0:
        print("要输入非负整数")
        return -1, 0
    numGuesses = 0
    low = 0.0
    # high = n
    high = max(n, 1.0)
    ans = (high + low)/2.0
    while abs(ans**2 - n) >= eps:
        if ans**2 < n:
            low = ans
        else:
            high = ans
        ans = (high + low)/2.0
        numGuesses += 1
        # print(low, high, ans, numGuesses, ans**2-n, eps)
        # input("按任意键继续")
    return ans, numGuesses
    
    
# 练习3，求方程的根
def fun(x):
    return x**3 + x**2 - 11
    
    
def findRoot(eps):
    lo, hi = -10, 10
    mid = (hi + lo)/2.0
    count = 0
    while abs(fun(mid)) > eps:
        if fun(lo)*fun(mid) < 0:
            hi = mid
        elif fun(mid)*fun(hi) < 0:
            lo = mid
        mid = (hi + lo)/2.0
        count += 1
        # print(lo, mid, hi, count, abs(fun(mid)))
        # input("按任意键继续")
    return mid


if __name__ == "__main__":
    n = int(input("输入一个完全平方数:"))
    res = findSquareRoot(n)
    res2, numGuesses = findSquareRoot2(n, 0.01, 0.001)
    res3, numGuesses3 = bisectionSearchForSquareRoot(n, 0.01)
    if res != -1:
        print(res,"*", res, "=", res**2)
    else:
        print("输入有误。")
    if res2 != -1:
        print(res2,"*", res2, "=", res2**2)
        print("猜测次数=", numGuesses)
    else:
        print("求解失败。")
    if res3 != -1:
        print(res3,"*", res3, "=", res3**2)
        print("猜测次数=", numGuesses3)
    else:
        print("求解失败。")
        
    print("线性查找:", Lsearch(Ls, 59))
    print("二分查找:", bsearch(Ls, 59))
    # 练习1
    res4, numGuesses4 = bisectionSearchForSquareRoot2(0.25, 0.01)
    print(res4)
    # 练习3
    print(findRoot(0.01))
    
    