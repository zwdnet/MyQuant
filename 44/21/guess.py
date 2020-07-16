# 《programming for the puzzled》实操
# 21.猜数字问题


# 封装二叉搜索树操作
class BSTVertex:
    def __init__(self, val, leftChild, rightChild):
        self.val = val
        self.leftChild = leftChild
        self.rightChild = rightChild
        
    def getVal(self):
        return self.val
        
    def getLeftChild(self):
        return self.leftChild
        
    def getRightChild(self):
        return self.rightChild
        
    def setVal(self, newVal):
        self.val = newVal
        
    def setLeftChild(self, newLeft):
        self.leftChild = newLeft
        
    def setRightChild(self, newRight):
        self.rightChild = newRight
        
        
class BSTree:
    def __init__(self, root = None):
        self.root = root
        
    def lookup(self, cVal):
        return self.__lookupHelper(cVal, self.root)
        
    def __lookupHelper(self, cVal, cVertex):
        if cVertex == None:
            return False
        elif cVal == cVertex.getVal():
            return True
        elif (cVal < cVertex.getVal()):
            return self.__lookupHelper(cVal, cVertex.getLeftChild())
        else:
            return self.__lookupHelper(cVal, cVertex.getRightChild())
            
    def insert(self, val):
        if self.root == None:
            self.root = BSTVertex(val, None, None)
        else:
            self.__insertHelper(val, self.root)
            
    def __insertHelper(self, val, pred):
        predLeft = pred.getLeftChild()
        predRight = pred.getRightChild()
        if (predRight == None and predLeft == None):
            if val < pred.getVal():
                pred.setLeftChild((BSTVertex(val, None, None)))
            else:
                pred.setRightChild((BSTVertex(val, None, None)))
        elif (val < pred.getVal()):
            if predLeft == None:
                pred.setLeftChild((BSTVertex(val, None, None)))
            else:
                self.__insertHelper(val, pred.getLeftChild())
        else:
            if predRight == None:
                pred.setRightChild((BSTVertex(val, None, None)))
            else:
                self.__insertHelper(val, pred.getRightChild())
                
    def inOrder(self):
        outputList = []
        return self.__inOrderHelper(self.root, outputList)
        
    def __inOrderHelper(self, vertex, outList):
        if vertex == None:
            return
        self.__inOrderHelper(vertex.getLeftChild(), outList)
        outList.append(vertex.getVal())
        self.__inOrderHelper(vertex.getRightChild(), outList)
        return outList
        
   
# 解决猜数字问题
def optimalBST(keys, prob):
    n = len(keys)
    opt = [[0 for i in range(n)] for j in range(n)]
    computeOptRecur(opt, 0, n-1, keys)
    tree = createBSTRecur(None, opt, 0, n-1, keys)
    print("平均最小猜测次数:", opt[0][n-1][0])
    printBST(tree.root)
    
    
def computeOptRecur(opt, left, right, prob):
    if left == right:
        opt[left][left] = (prob[left], left)
        return
    for r in range(left, right+1):
        if left <= r-1:
            computeOptRecur(opt, left, r-1, prob)
            leftval = opt[left][r-1]
        else:
            leftval = (0, -1)
        if r+1 <= right:
            computeOptRecur(opt, r+1, right, prob)
            rightval = opt[r+1][right]
        else:
            rightval = (0, -1)
        if r == left:
            bestval = leftval[0] + rightval[0]
            bestr = r
        elif bestval > leftval[0] + rightval[0]:
            bestr = r
            bestval = leftval[0] + rightval[0]
    weight = sum(prob[left:right+1])
    opt[left][right] = (bestval + weight, bestr)
    
    
def createBSTRecur(bst, opt, left, right, keys):
    if left == right:
        bst.insert(keys[left])
        return bst
    rindex = opt[left][right][1]
    rnum = keys[rindex]
    if bst == None:
        bst = BSTree(None)
    bst.insert(rnum)
    if left <= rindex-1:
        bst = createBSTRecur(bst, opt, left, rindex-1, keys)
    if rindex+1 <= right:
        bst = createBSTRecur(bst, opt, rindex+1, right, keys)
    return bst
    
    
def printBST(vertex):
    left = vertex.leftChild
    right = vertex.rightChild
    if left != None and right != None:
        print("值=", vertex.val, "左子节点=", left.val, "右子节点=", right.val)
        printBST(left)
        printBST(right)
    elif left != None and right == None:
        print("值=", vertex.val, "左子节点=", left.val, "右子节点=", "None")
        printBST(left)
    elif left == None and right != None:
        print("值=", vertex.val, "左子节点=", "None", "右子节点=", right.val)
        printBST(right)
    else:
        print("值=", vertex.val, "左子节点=", "None", "右子节点=", "None")
           


if __name__ == "__main__":
    # 测试二叉搜索树类
    root = BSTVertex(22, None, None)
    tree = BSTree(root)
    print(tree.lookup(22))
    tree.insert(25)
    tree.insert(35)
    tree.insert(9)
    print(tree.lookup(25))
    outres = tree.inOrder()
    print(outres)
    # 解决问题
    keys = [i+1 for i in range(7)]
    pr = [0.2, 0.1, 0.2, 0.0, 0.2, 0.1, 0.2]
    optimalBST(keys, pr)