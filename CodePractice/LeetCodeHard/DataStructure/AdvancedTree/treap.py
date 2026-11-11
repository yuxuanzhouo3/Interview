"""
11. Treap (BST + Heap priority / randomized BST)
Template: query / update / add / del
App: 隐式 treap 可做序列操作；竞赛替代 splay；期望 O(log n)。
"""

import random


class Node:
    def __init__(self, key, val=None):
        self.key = key
        self.val = val if val is not None else key
        self.prio = random.random()
        self.left = None
        self.right = None


def split(root, key):
    """< key | >= key"""
    if not root:
        return None, None
    if root.key < key:
        l, r = split(root.right, key)
        root.right = l
        return root, r
    l, r = split(root.left, key)
    root.left = r
    return l, root


def merge(a, b):
    if not a or not b:
        return a or b
    if a.prio > b.prio:
        a.right = merge(a.right, b)
        return a
    b.left = merge(a, b.left)
    return b


class Treap:
    def __init__(self):
        self.root = None

    def query(self, key):
        cur = self.root
        while cur:
            if key == cur.key:
                return cur.val
            cur = cur.left if key < cur.key else cur.right
        return None

    def update(self, key, val):
        cur = self.root
        while cur:
            if key == cur.key:
                cur.val = val
                return True
            cur = cur.left if key < cur.key else cur.right
        return False

    def add(self, key, val=None):
        if self.query(key) is not None:
            self.update(key, val if val is not None else key)
            return
        l, r = split(self.root, key)
        self.root = merge(merge(l, Node(key, val)), r)

    def delete(self, key):
        def rem(node):
            if not node:
                return None
            if key < node.key:
                node.left = rem(node.left)
            elif key > node.key:
                node.right = rem(node.right)
            else:
                return merge(node.left, node.right)
            return node

        self.root = rem(self.root)

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


def _test():
    random.seed(0)
    t = Treap()
    for k in [5, 2, 8, 1, 3]:
        t.add(k, k)
    assert t.inorder() == [1, 2, 3, 5, 8]
    assert t.query(3) == 3
    assert t.update(3, 30)
    t.delete(2)
    assert t.inorder() == [1, 3, 5, 8]
    print("Treap OK", t.inorder())


APPLICATION = """
- 随机优先级近似平衡，实现比 RB/AVL 短。
- 隐式 Treap: 用 size 当键，可切序列、区间反转（竞赛神器）。
- Java 无标准 Treap；TreeMap=RB；ConcurrentSkipListMap=跳表。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
