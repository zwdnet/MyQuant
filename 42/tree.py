# coding:utf-8
# 树的各种遍历方式


# 节点类
class Node:
    def __init__(self, item):
        self.item = item
        self.left = None
        self.right = None
        
    def __str__(self):
        return str(self.item)
        
        
# 二叉树类
class Tree:
    def __init__(self):
        self.root = Node("root")
        
    # 增加节点
    def add(self, item):
        node = Node(item)
        if self.root is None:
            self.root = node
        else:
            q = [self.root]
            
            while True:
                pop_node = q.pop(0)
                if pop_node.left is None:
                    pop_node.left = node
                    return
                elif pop_node.right is None:
                    pop_node.right = node
                    return
                else:
                    q.append(pop_node.left)
                    q.append(pop_node.right)
                    
    # 先序遍历
    def preorder(self, root):
        if root is None:
            return []
        result = [root.item]
        left_item = self.preorder(root.left)
        right_item = self.preorder(root.right)
        return result + left_item + right_item
        
    # 中序遍历
    def inorder(self, root):
        if root is None:
            return []
        result = [root.item]
        left_item = self.inorder(root.left)
        right_item = self.inorder(root.right)
        return left_item + result + right_item
        
    # 后序遍历
    def postorder(self, root):
        if root is None:
            return []
        result = [root.item]
        left_item = self.postorder(root.left)
        right_item = self.postorder(root.right)
        return left_item + right_item + result
        
    # 层次遍历
    def traverse(self):
        if self.root is None:
            return None
        q = [self.root]
        res = [self.root.item]
        while q != []:
            pop_node = q.pop(0)
            if pop_node.left is not None:
                q.append(pop_node.left)
                res.append(pop_node.left.item)
                
            if pop_node.right is not None:
                q.append(pop_node.right)
                res.append(pop_node.right.item)
        return res
            

if __name__ == "__main__":
    t = Tree()
    for i in range(10):
        t.add(i)
    print("前序遍历:", t.preorder(t.root))
    print("中序遍历:", t.inorder(t.root))
    print("后序遍历:", t.postorder(t.root))
    print("层次遍历:", t.traverse())
    
    