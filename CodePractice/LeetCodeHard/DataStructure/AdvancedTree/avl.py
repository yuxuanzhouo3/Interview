"""
2. AVL Tree (height-balanced BST)
Template: query / update / add / del
App: 读多写少的内存有序表；严格平衡 → 查询稳，旋转比红黑更频繁。
"""


class Node:
    def __init__(self, key, val=None):
        self.key = key
        self.val = val if val is not None else key
        self.left = None
        self.right = None
        self.h = 1


def _h(n):
    return n.h if n else 0


def _bf(n):
    return _h(n.left) - _h(n.right) if n else 0


def _fix_h(n):
    n.h = 1 + max(_h(n.left), _h(n.right))
    return n


def _rot_r(y):
    x, t2 = y.left, y.left.right
    x.right, y.left = y, t2
    _fix_h(y)
    _fix_h(x)
    return x


def _rot_l(x):
    y, t2 = x.right, x.right.left
    y.left, x.right = x, t2
    _fix_h(x)
    _fix_h(y)
    return y


class AVL:
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
        self.root = self._insert(self.root, key, val if val is not None else key)

    def _insert(self, node, key, val):
        if not node:
            return Node(key, val)
        if key < node.key:
            node.left = self._insert(node.left, key, val)
        elif key > node.key:
            node.right = self._insert(node.right, key, val)
        else:
            node.val = val
            return node
        return self._rebalance(_fix_h(node), key)

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
        return self._rebalance(_fix_h(node))

    def _rebalance(self, node, inserted_key=None):
        bf = _bf(node)
        if bf > 1:
            if inserted_key is not None and inserted_key > node.left.key:
                node.left = _rot_l(node.left)
            elif inserted_key is None and _bf(node.left) < 0:
                node.left = _rot_l(node.left)
            return _rot_r(node)
        if bf < -1:
            if inserted_key is not None and inserted_key < node.right.key:
                node.right = _rot_r(node.right)
            elif inserted_key is None and _bf(node.right) > 0:
                node.right = _rot_r(node.right)
            return _rot_l(node)
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


def _test():
    t = AVL()
    for k in range(1, 8):
        t.add(k, k)
    assert t.inorder() == list(range(1, 8))
    assert abs(_bf(t.root)) <= 1
    assert t.query(4) == 4
    assert t.update(4, 40) and t.query(4) == 40
    t.delete(4)
    assert t.query(4) is None
    assert t.inorder() == [1, 2, 3, 5, 6, 7]
    print("AVL OK", t.inorder(), "root_h=", t.root.h)


APPLICATION = """
- 读密集、内存有序字典；每次写可能多次旋转，写成本高于红黑树。
- Java TreeMap 选红黑而非 AVL：写路径旋转更少，工程更稳。
- 对比: BST 无平衡; AVL 严格 |bf|<=1; RB 近似平衡。
- LeetCode: 110 Balanced Binary Tree（判定）, 1382 Balance a BST.
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
