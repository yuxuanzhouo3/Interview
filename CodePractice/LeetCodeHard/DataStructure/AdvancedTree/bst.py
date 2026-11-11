"""
1. BST (Binary Search Tree)
Template: query / update / add / del
App: 教学/面试基线; Java TreeMap 底层不是朴素 BST，而是红黑树。
"""


class Node:
    def __init__(self, key, val=None):
        self.key = key
        self.val = val if val is not None else key
        self.left = None
        self.right = None


class BST:
    def __init__(self):
        self.root = None

    # ---------- query ----------
    def query(self, key):
        cur = self.root
        while cur:
            if key == cur.key:
                return cur.val
            cur = cur.left if key < cur.key else cur.right
        return None

    def contains(self, key):
        return self.query(key) is not None

    # ---------- update ----------
    def update(self, key, val):
        cur = self.root
        while cur:
            if key == cur.key:
                cur.val = val
                return True
            cur = cur.left if key < cur.key else cur.right
        return False

    # ---------- add ----------
    def add(self, key, val=None):
        if not self.root:
            self.root = Node(key, val)
            return
        cur, parent = self.root, None
        while cur:
            parent = cur
            if key == cur.key:
                cur.val = val if val is not None else key
                return
            cur = cur.left if key < cur.key else cur.right
        node = Node(key, val)
        if key < parent.key:
            parent.left = node
        else:
            parent.right = node

    # ---------- del ----------
    def delete(self, key):
        self.root = self._delete(self.root, key)

    def _delete(self, node, key):
        if not node:
            return None
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            if not node.left:
                return node.right
            if not node.right:
                return node.left
            succ = node.right
            while succ.left:
                succ = succ.left
            node.key, node.val = succ.key, succ.val
            node.right = self._delete(node.right, succ.key)
        return node

    def inorder(self):
        out = []

        def dfs(n):
            if not n:
                return
            dfs(n.left)
            out.append(n.key)
            dfs(n.right)

        dfs(self.root)
        return out


# ========== test case: query / update / add / del ==========
def _test():
    t = BST()
    for k in [5, 2, 8, 1, 3, 7, 9]:
        t.add(k, k * 10)
    assert t.query(3) == 30
    assert t.query(100) is None
    assert t.update(8, 88) and t.query(8) == 88
    assert not t.update(100, 1)
    t.add(6, 60)
    assert t.inorder() == [1, 2, 3, 5, 6, 7, 8, 9]
    t.delete(5)
    assert t.query(5) is None
    assert t.inorder() == [1, 2, 3, 6, 7, 8, 9]
    t.delete(1)
    t.delete(9)
    assert t.inorder() == [2, 3, 6, 7, 8]
    print("BST OK", t.inorder())


# ========== application (real use) ==========
APPLICATION = """
- 课堂/面试：有序集合的最简模型；最坏 O(n)（退化成链）。
- Java: HashMap/Hashtable = 哈希表 O(1) 均摊；TreeMap/TreeSet = 红黑树有序 Map，不是朴素 BST。
- 生产几乎不用裸 BST → 用 AVL / RedBlack / B+ / LSM 保证平衡或写放大可控。
- LeetCode: 98 Validate BST, 701 Insert, 450 Delete Node in BST.
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
