# coding:utf-8
# 《programming for the puzzled》实操
# 10.更多的皇后


# 迭代 欧几里得算法求最大公约数
def iGcd(m, n):
    while n > 0:
        m, n = n, m%n
    return m
    
    
# 递归版本求最大公约数
def rGcd(m, n):
    if m%n == 0:
        return n
    else:
        gcd = rGcd(n, m%n)
        return gcd
        
        
# 斐波那契数列
def rFib(x):
    if x == 0:
        return 0
    elif x == 1:
        return 1
    else:
        return rFib(x-1) + rFib(x-2)
        
        
# 迭代法计算斐波那契数列
def iFib(x):
    if x < 2:
        return x
    else:
        f, g = 0, 1
        for i in range(x-1):
            f, g = g, f+g
        return g
        
        
# 递归法解n皇后问题
def nQueens(size):
    board = [-1] * size
    rQueens(board, 0, size)
    print (board)
    displayBoard(board)


def noConflicts(board, current):
    for i in range(current):
        if (board[i] == board[current]):
            return False
        if (current - i == abs(board[current] - board[i])):
            return False
    return True 


def rQueens(board, current, size):
    if (current == size):
        return True
    else:
        for i in range(size):
            board[current] = i
            if (noConflicts(board, current)):
                done = rQueens(board, current + 1, size)
                if (done):
                    return True
        return False
        
     
# 画棋盘
def displayBoard(board):
    n = len(board)
    for i in range(n):
        for j in range(board[i]):
            print(".", end='')
        print("Q", end='')
        for j in range(board[i]+1, n):
            print(".", end='')
        print("\n")
            


if __name__ == "__main__":
    print(iGcd(24, 100))
    print(rGcd(24, 100))
    n = int(input("输入n值:"))
    for i in range(n):
        print(rFib(i))
    for i in range(n):
        print(iFib(i))
    nQueens(n)
    