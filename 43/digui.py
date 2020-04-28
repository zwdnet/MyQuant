# coding:utf-8
# 递归和回溯


# 力扣91题，解码的方法
def numDecodings(s):
    if s[0] == "0":
        return 0
    return decode(s, len(s) - 1)+1
    
    
def decode(chars, index):
    if index <= 1:
        return 1
        
    count = 0
    
    curr = chars[index]
    prev = chars[index-1]
    
    print("a", index, curr, prev)
    
    # 当前字符比0大，直接利用其之前的字符求得的结果
    if (curr > "0"):
        count = decode(chars, index - 1)
        print("b", count)
    # 数字在1-26之间才解码
    if (prev == "1" or (prev == "2" and curr <= "6")):
        count += decode(chars, index -2)
        print("c", count)
    
    return count


if __name__  == "__main__":
    s = "12"
    result = numDecodings(s)
    print(result)
    