# 《programming for the puzzled》实操
# 17.异位词问题


# 递归法
def anagramGrouping(input):
    output = []
    seen = [False] * len(input)
    for i in range(len(input)):
        if seen[i]:
            continue
        output.append(input[i])
        seen[i] = True
        for j in range(i+1, len(input)):
            if not seen[j] and anagram(input[i], input[j]):
                output.append(input[j])
                seen[j] = True
                
    return output


def anagram(str1, str2):
    return sorted(str1) == sorted(str2)
    
    
# 排序字母法
def anagramSortChar(input):
    canonical = []
    for i in range(len(input)):
        canonical.append((sorted(input[i]), input[i]))
    canonical.sort()
    output = []
    for t in canonical:
        output.append(t[1])
    return output
    
    
# 哈希算法
chToprime = {'a': 2, 'b': 3, 'c': 5, 'd': 7, 'e': 11, 'f': 13, 'g': 17, 'h': 19, 'i': 23, 'j': 29, 'k': 31, 'l': 37, 'm': 41, 'n': 43, 'o': 47, 'p': 53, 'q': 59, 'r': 61, 's': 67, 't': 71, 'u': 73, 'v': 79, 'w': 83, 'x': 89, 'y': 97, 'z': 101 }


def primeHash(str):
    if len(str) == 0:
        return 1
    else:
        return chToprime[str[0]] * primeHash(str[1:])
        
        
def anagramHash(input):
    output = sorted(input, key=primeHash)
    return output


if __name__ == "__main__":
    input = ["ate", "but", "eat", "tub", "tea"]
    output = anagramGrouping(input)
    print(output)
    output2 = anagramSortChar(input)
    print(output2)
    output3 = anagramHash(input)
    print(output3)
    