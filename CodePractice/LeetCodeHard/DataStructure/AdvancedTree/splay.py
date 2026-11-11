"""
12. Splay Tree
Template: query / update / add / del
App: 局部性好的访问序列；竞赛；部分 OS/缓存思想（把热点旋到根）。
"""


class Node:
    def __init__(self, key, val=None):
        self.key = key
        self.val = val if val is not None else key
        self.left = None
        self.right = None


class SplayTree:
    def __init__(self):
        self.root = None

    def _rot_r(self, x):
        y = x.left
        x.left = y.right
        y.right = x
        return y

    def _rot_l(self, x):
        y = x.right
        x.right = y.left
        y.left = x
        return y

    def _splay(self, root, key):
        if not root or root.key == key:
            return root
        if key < root.key:
            if not root.left:
                return root
            if key < root.left.key:
                root.left.left = self._splay(root.left.left, key)
                root = self._rot_r(root)
            elif key > root.left.key:
                root.left.right = self._splay(root.left.right, key)
                if root.left.right:
                    root.left = self._rot_l(root.left)
            return root if not root.left else self._rot_r(root)
        if not root.right:
            return root
        if key > root.right.key:
            root.right.right = self._splay(root.right.right, key)
            root = self._rot_l(root)
        elif key < root.right.key:
            root.right.left = self._splay(root.right.left, key)
            if root.right.left:
                root.right = self._rot_r(root.right)
        return root if not root.right else self._rot_l(root)

    def query(self, key):
        self.root = self._splay(self.root, key)
        if self.root and self.root.key == key:
            return self.root.val
        return None

    def update(self, key, val):
        if self.query(key) is None:
            return False
        self.root.val = val
        return True

    def add(self, key, val=None):
        val = key if val is None else val
        if not self.root:
            self.root = Node(key, val)
            return
        self.root = self._splay(self.root, key)
        if self.root.key == key:
            self.root.val = val
            return
        node = Node(key, val)
        if key < self.root.key:
            node.right = self.root
            node.left = self.root.left
            self.root.left = None
        else:
            node.left = self.root
            node.right = self.root.right
            self.root.right = None
        self.root = node

    def delete(self, key):
        if not self.root:
            return
        self.root = self._splay(self.root, key)
        if self.root.key != key:
            return
        if not self.root.left:
            self.root = self.root.right
        else:
            right = self.root.right
            self.root = self._splay(self.root.left, key)
            self.root.right = right

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
    t = SplayTree()
    for k in [5, 2, 8, 1, 3]:
        t.add(k)
    assert t.query(3) == 3
    assert t.update(3, 30) and t.query(3) == 30
    t.delete(2)
    assert t.inorder() == [1, 3, 5, 8]
    print("Splay OK", t.inorder())


APPLICATION = """
- 访问即旋转到根 → 热点路径摊还 O(log n)。
- 工程少见（并发难）；竞赛/理论常见。
- vs RB: RB 最坏更稳；Splay 常数/局部性有时更好。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
