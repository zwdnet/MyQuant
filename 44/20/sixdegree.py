# 《programming for the puzzled》实操
# 20.六度分离问题


small = {"A":["B", "C"],
                "B":["A", "C", "D"],
                "C":["A", "B", "E"],
                "D":["B", "E"],
                "E":["C", "D", "F"],
                "F":["E"]}
                
large = {'A': ['B', 'C', 'E'], 'B': ['A', 'C'],
                  'C': ['A', 'B', 'J'], 'D': ['E', 'F', 'G'],'E': ['A', 'D', 'K'], 'F': ['D', 'N'],'G': ['D', 'H', 'I'], 'H': ['G', 'M'],'I': ['G', 'P'], 'J': ['C', 'K', 'L'],'K': ['E', 'J', 'L'], 'L': ['J', 'K', 'S'],'M': ['H', 'N', 'O'], 'N': ['F', 'M', 'O'],'O': ['N', 'M', 'V'], 'P': ['I', 'Q', 'R'],'Q': ['P', 'W'], 'R': ['P', 'X'],'S': ['L', 'T', 'U'], 'T': ['S', 'U'],'U': ['S', 'T', 'V'], 'V': ['O', 'U', 'W'],'W': ['Q', 'V', 'Y'], 'X': ['R', 'Y', 'Z'],'Y': ['W', 'X', 'Z'], 'Z': ['X', 'Y']}
                
                
# 六度分离
def degreesOfSeparation(graph, start):
    if start not in graph:
        return -1
    visited = set()
    frontier = set()
    degrees = 0
    visited.add(start)
    frontier.add(start)
    while len(frontier) > 0:
        print(frontier, ":", degrees)
        degrees += 1
        newfront = set()
        for g in frontier:
            for next in graph[g]:
                if next not in visited:
                    visited.add(next)
                    newfront.add(next)
        frontier = newfront
    return degrees-1
    
    
# 计算一个图的分离度
def graphDegree(graph):
    vertices = graph.keys()
    maxDegree = degree = 0
    for v in vertices:
        degree = degreesOfSeparation(graph, v)
        if degree > maxDegree:
            maxDegree = degree
    return maxDegree


if __name__ == "__main__":
    # 测试集合操作
    frontier = {"A", "B", "D"}
    frontier.add("F")
    frontier.remove("A")
    print(frontier)
    # 广度优先遍历
    degreesOfSeparation(small, "A")
    degreesOfSeparation(large, "A")
    degreesOfSeparation(large, "U")
    # 计算图的分离度
    print("图的分离度为:%d"%graphDegree(large))
    