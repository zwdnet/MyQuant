# 《programming for the puzzled》实操
# 18.好记性问题


# 递归法
def coins(row, table):
    if len(row) == 0:
        table[0] = 0
        return 0, table
    elif len(row) == 1:
        table[1] = row[0]
        return row[0], table
    pick = coins(row[2:], table)[0] + row[0]
    skip = coins(row[1:], table)[0]
    result = max(pick, skip)
    table[len(row)] = result
    return result, table
    
    
def traceback(row, table):
    select = []
    i = 0
    while i < len(row):
        if (table[len(row)-i] == row[i]) or (table[len(row)-i] == table[len(row)-i-2] + row[i]):
            select.append(row[i])
            i += 2
        else:
            i += 1
    print("输入行:", row)
    print("表格:", table)
    print("选择的硬币:", select, "总数:", table[len(row)])


if __name__ == "__main__":
    row = [14, 3, 27, 4, 5, 15, 1]
    table = {}
    result, table = coins(row, table)
    print(result)
    print(table)
    traceback(row, table)
    