# 《programming for the puzzled》实操
# 19.周末聚会问题


# 判断图是否为二分图
def bipartiteGraphColor(graph, start, coloring, color):
    if start not in graph:
        return False, {}
        
    if start not in coloring:
        coloring[start] = color
    elif coloring[start] != color:
        return False, {}
    else:
        return True, coloring
        
    if color == 'Sha':
        newcolor = 'Hat'
    else:
        newcolor = 'Sha'
        
    for vertex in graph[start]:
        val, coloring = bipartiteGraphColor(graph, vertex, coloring, newcolor)
        if val == False:
            return False, {}
        
    return True, coloring


if __name__ == "__main__":
    dangling = {"A":["B", "E"],
                         "B":["A", "E", "C"],
                         "C":["B", "D"],
                         "D":["C", "E"],
                         "E":["A", "B", "D"]}
    res, col = bipartiteGraphColor(dangling, "A", {}, "Sha")
    print(res, col)
    