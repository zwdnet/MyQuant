# coding:utf-8
# 《programming for the puzzled》实操
# 5.打碎水晶


def howHardIsTheCrystal(n, d):
    r = 1
    while (r**d < n):
        r += 1
    print("选择的基数为", r)
    # 练习1，有时d太大，会跳过第一个球
    # 减少d的值
    newd = d
    while (r**(newd-1) > n):
        newd -= 1
    if newd < d:
        print("只用了", newd, "个球")
    d = newd
    
    numDrops = 0
    # 练习2 输出坏了的球
    numBreaks = 0
    # 练习3，输出正在考虑的楼层区间
    start = 0
    end = n
    floorNoBreak = [0]*d
    for i in range(d):
        for j in range(r-1):
            # 练习3，输出正在考虑的区间
            print("正在考虑", start, "到", end, "的楼层。")
            floorNoBreak[i] += 1
            Floor = convertToDecimal(r,  d, floorNoBreak)
            if Floor > n:
                floorNoBreak[i] -= 1
                break
            print("从", Floor, "层扔下第", i+1, "个球。")
            yes = input("水晶球裂了吗?(yes/no):")
            numDrops += 1
            if yes == "yes":
                floorNoBreak[i] -= 1
                end = Floor-1
                break
            # 练习2
            else:
                numBreaks += 1
                start = Floor+1
                
    hardness = convertToDecimal(r,  d, floorNoBreak)
    print("硬度为:", hardness)
    print("扔球的总数为:", numDrops)
    # 练习2
    print("扔坏了的球的个数为:", numBreaks)
    
    return
    
    
def convertToDecimal(r,  d, rep):
    number = 0
    for i in range(d-1):
        number = (number + rep[i])*r
    number += rep[d-1]
    
    return number


if __name__ == "__main__":
    howHardIsTheCrystal(128, 6)
    