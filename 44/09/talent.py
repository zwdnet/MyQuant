# coding:utf-8
# 《programming for the puzzled》实操
# 9.脱口秀节目


Talent = ["Sing", "Dance", "Magic", "Act", "Flex", "Code"]
Candidates = ["Aly", "Bob", "Cal", "Don", "Eve", "Fay"]
CandidateTalents = [["Flex", "Code"], 
                                    ["Dance", "Magic"],
                                    ["Sing", "Magic"],
                                    ["Dance", "Act", "Code"],
                                    ["Act", "Code"]]
                                    
                                 
# 生成所有候选人组合
def Hire4Show(candList, candTalents, talentList):
    n = len(candList)
    hire = candList[:]
    for i in range(2**n):
        Combination = []
        num = i
        for j in range(n):
            if (num % 2 == 1):
                Combination = [candList[n-1-j]] + Combination
            num = nun//2
            
                

                                    
if __name__ == "__main__":
    pass
