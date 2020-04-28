# coding:utf-8
# 哈希算法


# 方法一，双指针法
def twoSum(nums, target):
    res = []
    newnums = nums[:]
    newnums.sort()
    left = 0
    right = len(newnums) - 1
    while left < right:
        if newnums[left] + newnums[right] == target:
            for i in range(0, len(nums)):
                if nums[i] == nums[left]:
                    res.append(i)
                    break
            for i in range(len(nums)-1, -1, -1):
                if nums[i] == nums[right]:
                    res.append(i)
                    break
            res.sort()
            break
        elif newnums[left] + newnums[right] < target:
            left += 1
        elif newnums[left] + newnums[right] > target:
            right -= 1
    return (res[0]+1, res[1]+1)
    
   
# 方法二，哈希算法
def twoSum2(nums, target):
    dict = {}
    for i in range(len(nums)):
        m = nums[i]
        if target-m in dict:
            return (dict[target-m]+1, i+1)
        dict[m] = i
        

# 字符串模式匹配
def wordPattern(pattern, input):
    word = input.split(' ')
    if len(word) != len(pattern):
        return False
    hash = {}
    used = {}
    for i in range(len(pattern)):
        if pattern[i] in hash:
            if hash[pattern[i]] != word[i]:
                return False
        else:
            if word[i] in used:
                return False
            hash[pattern[i]] = word[i]
            used[word[i]] = True
    return True


if __name__ == "__main__":
    # 给定数组，和一个目标值，找出数组中和为目标值的两个数字
    nums = [3, 4, 5, 7, 10]
    target = 11
    res = twoSum(nums, target)
    print(res)
    print(nums[res[0]-1], nums[res[1]-1])
    
    # 给定一个模式和一组字符串，看二者模式是否一致
    pattern = "一 二 二 一"
    input = "平 安 安 平"
    print(wordPattern(pattern, input))
    
    
    