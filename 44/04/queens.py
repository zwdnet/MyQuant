# coding:utf-8
# 《programming for the puzzled》实操
# 4.N皇后问题


import numpy as np


# 工具函数 输出棋盘
def displayBoard(board):
    n = len(board)
    for i in range(n):
        print(board[i])
        
        
# 工具函数，判断放置皇后的情况是否符合要求
def Judge(board, col, row, n):
    
    return True
    
    
# 工具函数，将棋盘数据转换成
    

# 穷举法
def QJ(n = 5):
    if n != 5:
        print("输入有误!穷举法只用于n=5的情况")
        return
    board = [[0]*n]*n  
    displayBoard(board)
    # 穷举每一种情况
    ans = []
    times = 0
            
    for i1 in range(n):
        board[i1][0] = 1
        for i2 in range(n):
            board[i2][1] = 1
            if Judge(board, 1, i2, n):
                for i3 in range(n):
                    board[i3][2] = 1
                    if Judge(board, 2, i3, n):
                        for i4 in range(n):
                            board[i4][3] = 1
                            if Judge(board, 3, i4, n):
                                for i5 in range(n):
                                    board[i5][4] = 1
                                    # 判断是否符合要求
                                    if Judge(board, 4, i5, n):
                                        ans.append(board)
                                    times += 1
                                    board[i5][4] = 0
                            board[i4][3] = 0
                    board[i3][2] = 0
            board[i2][1] = 0
        board[i1][0] = 0
    return ans, times
    
    
# 判断布局是否违反规则    
def noConflicts2(board, current):
    for i in range(current):
        # 每行一个
        if (board[i] == board[current]):
            return False
        # 对角线
        if (current - i == abs(board[current] - board[i])):
            return False
    print("c")
    return True
        
        
# 八皇后 回溯
def EightQueens(n=8):
    board = [-1]*n
    count = 0
    for i in range(n):
        board[0] = i
        for j in range(n):
            board[1] = j
            if not noConflicts2(board, 1):
                continue
            for k in range(n):
                board[2] = k
                if not noConflicts2(board, 2):
                    continue
                for l in range(n):
                    board[3] = l
                    if not noConflicts2(board, 3):
                        continue
                    for m in range(n):
                        board[4] = m
                        if not noConflicts2(board, 4):
                            continue
                        for o in range(n):
                            board[5] = o
                            if not noConflicts2(board, 5):
                                continue
                            for p in range(n):
                                board[6] = p
                                if not noConflicts2(board, 6):
                                    continue
                                for q in range(n):
                                    board[7] = q
                                    if noConflicts2(board, 7):
                                        print(board)
                                        count += 1
    return count
        
        
# 八皇后 回溯
def EightQueens2(board, n=8):
    # board = [-1]*n
    count = 0
    for i in range(n):
        if board[0] == -1:
            board[0] = i
        for j in range(n):
            if board[1] == -1:
                board[1] = j
            print(board)
            if not noConflicts2(board, 1):
                continue
            for k in range(n):
                if board[2] == -1:
                    board[2] = k
                if not noConflicts2(board, 2):
                    continue
                for l in range(n):
                    if board[3] == -1:
                        board[3] = l
                    if not noConflicts2(board, 3):
                        continue
                    for m in range(n):
                        if board[4] == -1:
                            board[4] = m
                        if not noConflicts2(board, 4):
                            continue
                        for o in range(n):
                            if board[5] == -1:
                                board[5] = o
                            if not noConflicts2(board, 5):
                                continue
                            for p in range(n):
                                if board[6] == -1:
                                    board[6] = p
                                if not noConflicts2(board, 6):
                                    continue
                                for q in range(n):
                                    print(board)
                                    if board[7] == -1:
                                        board[7] = q
                                    if noConflicts2(board, 7):
                                        print(board)
                                        count += 1
    return count


if __name__ == "__main__":
    n = int(input("输入问题规模:(>0)"))
    res, times = QJ()
    print("答案总数:", len(res), ",运算次数:", times)
    res = EightQueens()
    print("八皇后问题，答案总数:", res)
    board = [-1, 4, -1, -1, -1, -1, -1, 0]
    res = EightQueens2(board)
    print("带条件的八皇后问题，答案总数:", res)
    