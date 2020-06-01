# coding:utf-8
# 《programming for the puzzled》实操
# 11.瓷砖铺地


# 分治法，归并排序
def mergeSort(L):
    # 不加下面这行递归无法结束
    if len(L) < 2:
        return L
    if len(L) == 2:
        if L[0] <= L[1]:
            return [L[0], L[1]]
        else:
            return [L[1], L[0]]
    else:
        middle = len(L)//2
        left = mergeSort(L[:middle])
        right = mergeSort(L[middle:])
        return merge(left, right)
        
        
# 合并函数
def merge(left, right):
    result = []
    i,j = 0,0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    while i < len(left):
        result.append(left[i])
        i += 1
    while j < len(right):
        result.append(right[j])
        j += 1
    return result
    
    
# 用递归法铺砖
EMPTYPIECE = -1

def recursiveTile(yard, size, originR, originC, rMiss, cMiss, nextPiece):
    quadMiss = 2*(rMiss >= size // 2) + (cMiss >= size // 2)
    if size == 2:
        piecePos = [(0,0), (0,1), (1,0), (1,1)]
        piecePos.pop(quadMiss)
        for (r, c) in piecePos:
            yard[originR+r][originC+c] = nextPiece
        nextPiece = nextPiece + 1
        return nextPiece
        
    for quad in range(4):
        shiftR = size//2*(quad >= 2)
        shiftC = size//2*(quad % 2 == 1)
        if quad == quadMiss:
            nextPiece = recursiveTile(yard, size//2, originR+shiftR, originC+shiftC, rMiss-shiftR, cMiss-shiftC, nextPiece)
        else:
            newrMiss = (size//2-1)*(quad<2)
            newcMiss = (size//2-1)*(quad%2 == 0)
            nextPiece = recursiveTile(yard, size//2, originR+shiftR, originC+shiftC, newrMiss, newcMiss, nextPiece)
        centerPos = [(r + size//2 - 1, c + size//2 - 1) for (r,c) in [(0,0), (0,1), (1,0), (1,1)]]
    centerPos.pop(quadMiss)
    for (r,c) in centerPos: 
        yard[originR + r][originC + c] = nextPiece
    nextPiece = nextPiece + 1

    return nextPiece
    
    
def tileMissingYard(n, rMiss, cMiss):
    yard = [[EMPTYPIECE for i in range(2**n)] for j in range(2**n)]
    recursiveTile(yard, 2**n, 0, 0, rMiss, cMiss, 0)
    return yard
    

def printYard(yard):
    for i in range(len(yard)):
        row = ""
        for j in range(len(yard[0])):
            if yard[i][j] != EMPTYPIECE:
                row += chr((yard[i][j] % 26) + ord("A"))
            else:
                row += " "
        print(row)


if __name__ == "__main__":
    L = [2, 4, 3, 9, 7, 8, 6, 4, 1, 5, 7, 3]
    res = mergeSort(L)
    print(L, "\n", res)
    printYard(tileMissingYard(3,4,6))
    