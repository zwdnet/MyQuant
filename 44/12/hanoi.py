# coding:utf-8
# 《programming for the puzzled》实操
# 12.汉诺塔


def hanoi(numRings, startPeg, endPeg):
    numMoves = 0
    if numRings == 1:
        print("从", startPeg, "到", 6-startPeg-endPeg, "移动", numRings, "号盘子。")
        print("从", 6-startPeg-endPeg, "到",startPeg , "移动", numRings, "号盘子。")
        numMoves += 2
    else:
        print("从", startPeg, "到", endPeg, "移动", numRings, "号盘子。")
        numMoves += hanoi(numRings-1, startPeg, 6-startPeg-endPeg)
        print("从", startPeg, "到", endPeg, "移动", numRings, "号盘子。")
        numMoves += 1
        numMoves += hanoi(numRings-1, 6-startPeg-endPeg, startPeg)
    return numMoves


if __name__ == "__main__":
    numMoves = hanoi(8, 1, 3)
    print("步数等于:", numMoves)
