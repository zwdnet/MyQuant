# coding:utf-8
# 《programming for the puzzled》实操
# 9.脱口秀节目


Talents = ["Sing", "Dance", "Magic", "Act", "Flex", "Code"]
Candidates = ["Aly", "Bob", "Cal", "Don", "Eve", "Fay"]
CandidateTalents = [ ['Flex', 'Code'], ['Dance', 'Magic'], ['Sing', 'Magic'], ['Sing', 'Dance'], ['Dance', 'Act', 'Code'], ['Act', 'Code'] ]
                                    
                                 
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
            num = num // 2

        if Good(Combination, candList, candTalents, talentList):
            if len(hire) > len(Combination):
                hire = Combination

    print ('Optimum Solution:', hire)
            
                
def Good(Comb, candList, candTalents, AllTalents):
    for tal in AllTalents:
        cover = False
        for cand in Comb:
            candTal = candTalents[candList.index(cand)]
            if tal in candTal:
                cover = True
        if not cover:
            return False 
    return True

                                    
if __name__ == "__main__":
    Hire4Show(Candidates, CandidateTalents, Talents)
    
    ST2 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    CL2 = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    CT2 = [ [4, 5, 7], [1, 2, 8], [2, 4, 6, 9], [3, 6, 9], [2, 3, 9], [7, 8, 9], [1, 3, 7] ]
    Hire4Show(CL2, CT2, ST2)
