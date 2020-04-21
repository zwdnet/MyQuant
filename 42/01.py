# coding:utf-8
# 拉钩课程《300分钟搞定数据结构与算法》练习


from string import *
import collections


# 1.翻转字符串
def reverse(str):
    head = 0
    tail = len(str) - 1
    ls = ["a"]*len(str)
    while head <= tail:
        ls[head] = str[tail]
        ls[tail] = str[head]
        head += 1
        tail -= 1
    result = ''.join(ls)
    return result
    
    
# 2.两个字符串s和t，判断t是否是s的异位词。
def isAnagram(s, t):
    alpha = [chr(i) for i in range(97, 123)]
    alphadic = {char:0 for char in alpha}
    l1 = len(s)
    l2 = len(t)
    if l1 != l2:
        return False
    for i in range(l1):
        alphadic[s[i]] += 1
        alphadic[t[i]] -= 1
    for i in range(97, 123):
        if alphadic[chr(i)] != 0:
            return False
    return True
    
    
# 3.每k个节点翻转链表
# 定义单链表节点
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
        self.length = 1
        
    # 插入节点
    def append(self, item):
        node = ListNode(item)
        if self.next == None:
            self.next = node
        else:
            cur = self.next
            while cur.next is not None:
                cur = cur.next
            cur.next = node
        self.length += 1
        
    # 输出单链表
    def displayList(self, head):
        while head != None:
            print(head.val)
            head = head.next
            
    # 获取链表长度
    def getLength(self):
        return self.length

        
def reverseKGroup(head, k):
#    prev = None
#    cur = head
#    length = head.getLength()
#    for i in range(k):
#        next = cur.next
#        cur.next = prev
#        prev = cur
#        cur = next
#    return cur
    h = ListNode(-1)
    h.next = head
    cur = pre = h
    
    n = -1
    while cur != None:
        n += 1
        cur = cur.next
        
    while n >= k:
        cur = pre.next
        for _ in range(k - 1):
            lat = cur.next
            cur.next = lat.next
            lat.next = pre.next
            pre.next = lat
        pre = cur
        n -= k
        
    return h.next
    

# 4.两两交换链表节点
def swapPairs(head):
    k = 2
    h = ListNode(-1)
    h.next = head
    cur = pre = h
    
    n = -1
    while cur != None:
        n += 1
        cur = cur.next
        
    while n >= k:
        cur = pre.next
        for _ in range(k - 1):
            lat = cur.next
            cur.next = lat.next
            lat.next = pre.next
            pre.next = lat
        pre = cur
        n -= k
        
    return h.next
    
    
# 5.判断括号符号的有效性
class Stack:
    def __init__(self):
        self.data = []
        self.length = 0
        
    def push(self, val):
        self.data.append(val)
        self.length += 1
        
    def pop(self):
        if self.length == 0:
            return None
        res = self.data.pop()
        self.length -= 1
        return res
        
    def getLength(self):
        return self.length
    
    def get_item(self):
        if self.length == 0:
            return None
        else:
            return self.data[self.length-1]
        
    
def isValid(s):
    length = len(s)
    if length == 0:
        return True
    if length % 2 != 0:
        return False
    stack = Stack()
    left = ["(", "[", "{"]
    right =  [")", "]", "}"]
    for i in range(length):
        c = s[i]
        cp = None
        if c in left:
            stack.push(c)
        elif c in right:
            cp = stack.pop()
            if ((c == ")" and cp != "(") or
                 (c == "]" and cp != "[") or
                 (c == "}" and cp != "{")):
                return False

    if stack.getLength() == 0:
        return True
    else:
        return False
        
        
# ⑥气温列表
# 算法一，穷举，超时了
def dailyTemperatures(T):
    n = len(T)
    result = [0]*n
    for i in range(n):
        T0 = T[i]
        d = 1
        bHigh = False
        for j in range(i+1, n):
            if T[j] > T[i]:
                bHigh = True
                break
            d += 1
        if bHigh == False:
            d = 0
        result[i] = d
    return result
    

# 算法2，用堆栈，降低时间复杂度
def dailyTemperatures2(T):
#    print(T)
#    n = len(T)
#    result = [0]*n
#    stack = Stack()
#    for i in range(n):
#        if stack.getLength() == 0:
#            stack.push(i)
#            T_top = T[i]
#            continue
#        top_index = stack.get_item()
#        if T[i] > T[top_index]:
#            i_top = stack.pop()
#            result[i_top] = i - i_top
#            stack.push(i)
#        else:
#            stack.push(i)
#        print(i, T[i], top_index, T[top_index])
#    return result
    ans = [0]*len(T)
    stack = []
    for i in range(len(T) - 1, -1, -1):
        while stack and T[i] >= T[stack[-1]]:
            stack.pop()
        if stack:
            ans[i] = stack[-1] - i
        stack.append(i)
    return ans
    
    
# 7.基本计算器
def calculate(s):
    stack = []
    res = 0
    num = 0
    sign = 1
    for c in s:
        if c.isdigit():
            num = num*10 + int(c)
        elif c == "+" or c == "-":
            res = res + num*sign
            if c == "+":
                sign = 1
            else:
                sign = -1
            num = 0
        elif c == "(":
            stack.append(res)
            stack.append(sign)
            sign = 1
            res = 0
        elif c == ")":
            res = res + sign*num
            old_sign = stack.pop()
            old_res = stack.pop()
            res = old_res + old_sign*res
            sign = 1
            num = 0
    res = res + sign*num
    return res
    

# 8.高级计算器
def highCulate(s):
    # 初始化sign为 “+”，是因为开头是数字
    num ,stack ,sign = 0 , [] , '+'
    for i in range(len(s)):
        ch = s[i]
        if ch.isdigit():
            num = num * 10 + int(ch) #根据当前数字之前的符号，来决定如何处理当前数值
        # 并将处理后的数值压入栈中
        if ch in "+-*/" or i == len(s)-1:
            if sign == "+" :
                stack.append(num)
            elif sign == "-" :
                stack.append(-num)
            elif sign == "*":
                stack.append(stack.pop() * num)
            else:
                stack.append(int(stack.pop()/num))
            num = 0
            sign = ch
    return sum(stack)
    

# 9.柱状图最大矩形
def maxJu(heights):
    stack = []
    maxArea = 0
    for i in range(len(heights)):
        while stack and heights[i] < heights[stack[-1]]:
            top = stack.pop()
            wide = 0
            if stack:
                wide = i - stack[-1] -1
            else:
                wide = i
            maxArea = max(maxArea, heights[top] * wide)
        stack.append(i)
    return maxArea
    
    
# 滑动窗口最大值
def maxSlidingWindow(nums, k):
    i = 0
    n = len(nums)
    res = [0]*(n-k+1)
    while i <= n - k:
        temp = nums[i:i+k]
        res[i] = max(temp)
        i += 1
    return res
    

# 用双端队列求解
def maxSlidingWindow2(nums, k):
    deque = collections.deque()
    res = []
    for i, num in enumerate(nums):
        while deque and deque[0] <= i - k: 
            deque.popleft()
        while deque and num > nums[deque[-1]]:
            deque.pop()
        deque.append(i)
        if i >= k-1:
            res.append(nums[deque[0]])
    return res
    

if __name__ == "__main__":
    # 1.翻转字符串
    str = "algorithm"
    print(reverse(str))
    
    # 2.两个字符串s和t，判断t是否是s的异位词。
    result = isAnagram("aaabbb", "ababab")
    if (result == True):
        print("是异位词")
    else:
        print("不是异位词")
        
    # 3. 翻转链表的k个节点
    
    head = ListNode(None)
    for i in range(1, 10):
        head.append(i)
    head.displayList(head)
    print("链表长度:{}".format(head.getLength()))
    head =  reverseKGroup(head, 3)   
    head.displayList(head)
    head = swapPairs(head)
    head.displayList(head)
    
    # 4.符号有效性判断
    s = "()"
    stack = Stack()
    for i in range(len(s)):
        stack.push(s[i])
    print("栈的长度:{}".format(stack.getLength()))
    print(isValid(s))
    
    # 5.气温列表问题
    t = [73, 74, 75, 71, 69, 72, 76, 73]
    result = dailyTemperatures(t)
    print(result)
    result2 = dailyTemperatures2(t)
    print(result2)
    
    # 6.基本计算器
    s = "1+(2-5)-2"
    res = calculate(s)
    print(res)
    
    # 7.增加乘除法的计算器
    s = "3/2"
    res = highCulate(s)
    print(res)
    
    # 8.柱状图的最大矩形面积
    heights = [2, 1, 5, 6, 2, 3]
    maxArea = maxJu(heights)
    print(maxArea)
    
    # 9.最大窗口值
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    res = maxSlidingWindow(nums, k)
    print(res)
    
    res = maxSlidingWindow2(nums, k)
    print(res)
    