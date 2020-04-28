# coding:utf-8
# 回溯算法


from datetime import datetime
import numpy as np
import pandas as pd


# 排列问题
def permute(nums):
    results = []
    track = []
    backtrack(nums, track, results)
    return results
    
    
def backtrack(nums, track, results):
    if len(track) == len(nums):
        results.append(tuple(track))
        return
        
    for i in range(len(nums)):
        if nums[i] in track:
            continue
        # 选择
        track.append(nums[i])
        # 进入下一层决策树
        backtrack(nums, track, results)
        # 取消选择
        track.pop()
        
        
# N皇后问题
def  solveNQ(N):
    board = []
    for i in range(N):
        board.append([0]*N)
    
    results = []
    global n
    n = 0
    solveNQUtil(board, 0, N, results)
    # printSolution(results, N)
    return True
    
    
def printSolution(board, N):
    global n
    n += 1
    print("\n第{}个结果。\n".format(n))
    for i in range(N):
        for j in range(N):
            print(board[i][j], end = " ")
        print("\n")
        
        
def solveNQUtil(board, col, N, results):
    if col >= N:
        printSolution(board, N)
        return
        
    for i in range(N):
        if isSafe(board, i, col, N):
            board[i][col] = 1
            solveNQUtil(board, col+1, N, results)
            board[i][col] = 0
    
    
def isSafe(board, row, col, N):
    # 左侧的列
    for i in range(col):
        if board[row][i] == 1:
            return False
            
    # 左上对角线
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
            
    # 左下对角线
    for i, j in zip(range(row, N, 1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
            
    return True
    
    
# 数独的回溯解法
class sovSudoku:
    def __init__(self, board = []):
        self._b = board.copy()
        self._t = 0
        self._n = 0
        
    # 主循环，尝试x,y处的解答
    def trysxy(self, x, y):
        self._n += 1
        if self._b[x][y] == 0:
            pv = self.getPrem(x, y)
            for v in pv:
                self._t += 1
                if self.checkNotSame(x, y, v):
                    self._b[x][y] = v
                    nx, ny = self.getNext(x, y)
                    if nx == -1:
                        return True
                    else:
                        _end = self.trysxy(nx, ny)
                        if not _end:
                            self._b[x][y] = 0
                        else:
                            return True
            
    # 得到x, y处可以填的值
    def getPrem(self, x, y):
        prem = []
        rows = list(self._b[x])
        rows.extend([self._b[i][y] for i in range(9)])
        cols = set(rows)
        for i in range(1, 10):
            if i not in cols:
                prem.append(i)
        return prem
        
    # 检查每行每列和每个宫内是否有与b(x, y)相同的数字
    def checkNotSame(self, x, y, val):
        # 第x行
        for row_item in self._b[x]:
            if row_item == val:
                return False
        # 第y列
        for rows in self._b:
            if rows[y] == val:
                return False
        ax = x//3*3
        ab = y//3*3
        for r in range(ax, ax+3):
            for c in range(ab, ab+3):
                if self._b[r][c] == val:
                    return False
        return True
        
    # 得到下一个未填项
    def getNext(self, x, y):
        for ny in range(y+1, 9):
            if self._b[x][ny] == 0:
                return (x, ny)
        for row in range(x+1, 9):
            for ny in range(0, 9):
                if self._b[row][ny] == 0:
                    return (row, ny)
        return (-1, -1)
        
    def solve(self):
        if self._b[0][0] == 0:
            self.trysxy(0, 0)
        else:
            x, y = self.getNext(0, 0)
            self.trysxy(x, y)
            
            def updateSudo(self, cb):
                if len(cb) == 9 and len(cb[0] == 9):
                    self._b = cb
                else:
                    print("错误结果", len(cb), len(cb[0]))
                   
    def __str__(self):
        return '{0}{1}{2}'.format('[',',\n'.join([str(i) for i in self._b]),']')
        
    # 获得回溯次数
    def getTNum(self):
        return self._n


if __name__ == "__main__":
    # 全排列问题
    nums = [1, 2, 3]
    result = permute(nums)
    print(result)
    # N皇后问题
    solveNQ(8)
    # 数独问题
    s1 = [
            [8,0,0, 0,0,0, 0,0,0],
            [0,0,3, 6,0,0, 0,0,0],
            [0,7,0, 0,9,0, 2,0,0],
            [0,5,0, 0,0,7, 0,0,0],
            [0,0,0, 0,4,5, 7,0,0],
            [0,0,0, 1,0,0, 0,3,0],
            [0,0,1, 0,0,0, 0,6,8],
            [0,0,8, 5,0,0, 0,1,0],
            [0,9,0, 0,0,0, 4,0,0]
    ]
    begin = datetime.now()
    ss = sovSudoku(s1)
    ss.solve()
    print(datetime.now() - begin)
    print(ss)
    print(ss.getTNum())
    
    m = [
        [6, 0, 0, 1, 0, 0, 7, 0, 8],
        [0, 0, 0, 8, 0, 0, 2, 0, 0],
        [2, 3, 8, 0, 5, 0, 1, 0, 0],
        [0, 0, 0, 0, 4, 0, 0, 9, 2],
        [0, 0, 4, 3, 0, 8, 6, 0, 0],
        [3, 7, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 3, 0, 7, 0, 5, 2, 6],
        [0, 0, 2, 0, 0, 4, 0, 0, 0],
        [9, 0, 7, 0, 0, 6, 0, 0, 4]
    ]
    begin = datetime.now()
    ss = sovSudoku(m
    )
    ss.solve()
    print(datetime.now() - begin)
    print(ss)
    print(ss.getTNum())
    # 背包问题
    bestV = 0
    curW = 0
    curV = 0
    bestx = None
    
    def backtrack(i):
        global bestV,curW,curV,x,bestx
        if i >= n:
            if bestV < curV:
                bestV = curV
                bestx = x[:]
        else:
            if curW + w[i] <= c:
                x[i] = True
                curW += w[i]
                curV += v[i]
                backtrack(i+1)
                curW -= w[i]
                curV -= v[i]
            x[i] = False
            backtrack(i+1)

    n=5
    c=10
    w=[2,2,6,5,4]
    v=[6,3,5,4,6]
    x=[False for i in range(n)]
    backtrack(0)
    print(bestV)
    print(bestx)
    