# 《programming for the puzzled》实操
# 15.计数问题


def makeChange(bills, target, sol = []):
    if sum(sol) == target:
        print(sol)
        return
        
    if sum(sol) > target:
        return
        
    for bill in bills:
        newSol = sol[:]
        newSol.append(bill)
        makeChange(bills, target, newSol)
    return
    
    
# 去除重复结果
def makeSmartChange(bills, target, highest, sol = []):
    if sum(sol) == target:
        print(sol)
        return
        
    if sum(sol) > target:
        return
    
    for bill in bills:
        if bill >= highest:
            newSol = sol[:]
            newSol.append(bill)
            # 就这里不一样
            makeSmartChange(bills, target, bill, newSol)
    return


if __name__ == "__main__":
    bills = [1, 2, 5]
    makeChange(bills, 6)
    print("排除重复答案")
    makeSmartChange(bills, 6, 1)
    