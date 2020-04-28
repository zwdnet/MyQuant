# coding:utf-8
# 优先队列
from queue import PriorityQueue


def kthelem(data, k):
    pq = PriorityQueue()
    for i in data:
        pq.put(i)
    for i in range(k):
        result = pq.get()
    return result
    
    
# 力扣347题，找出给定序列中出现频率最高的前k个元素
def kthfreq(nums, k):
    # 用哈希表存储数据出现的频率
    freq = dict()
    for i in nums:
        freq[i] = 0
    for i in nums:
        freq[i] += 1
    # 用优先队列找出频率前k的数据
    pq = PriorityQueue()
    for key in freq.keys():
        pq.put([-freq[key], key])
    results = [0]*k
    for i in range(k):
        results[i] = pq.get()[1]
    return results


if __name__ == "__main__":
    # 测试优先队列的使用
    pq = PriorityQueue()
    for i in range(3, 0, -1):
        pq.put(i)
    while not pq.empty():
        print(pq.get())
        
    # 找出序列中第k大的元素
    data = [8, 9, 3, 6, 7, 5, 54, 33, 65, 90]
    k = 5
    result = kthelem(data, k)
    print(data)
    print("第{}大的数为:{}".format(k, result))
    
    # 力扣347题，找出给定序列中出现频率最高的前k个元素
    data = [1,1,2,2,1,3,3,4,3,4,6,3]
    k = 3
    result = kthfreq(data, k)
    print(result)
    

    