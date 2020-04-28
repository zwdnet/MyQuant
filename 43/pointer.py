# coding:utf-8
# 指针


# 合并两个数组
def merge(arr1, arr2):
    ind = 0
    ans = arr1.copy()
    for i in range(0, len(arr2)):
        while ind < len(arr1):
            if arr2[i] <= arr1[ind]:
                ans.insert(ind+i, arr2[i])
                break
            else:
                ind += 1
        else:
            ans = ans + arr2[i:]
    return ans
                


if __name__ == "__main__":
    arr1 = [1, 3, 4, 6, 10]
    arr2 = [2, 5, 8, 11]
    arr = merge(arr1, arr2)
    print(arr)
    # 二分查找
    nums = [1,3,5,6,7,8,13,14,15,17,18,24,30,43,56]
    head, tail = 0, len(nums)
    search = int(input("输入要查询的数字:"))
    
    ans = -1
    while tail - head > 1:
        mid = (head + tail) // 2
        if search < nums[mid]:
            tail = mid
        elif search > nums[mid]:
            head = mid+1
        else:
            ans = mid
            break
    else:
        if search == nums[head]:
            ans = head
        else:
            ans = -1
    print(ans)