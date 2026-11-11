"""
4. B-Tree (multi-way balanced search tree)
Template: query / update / add / del
App: 文件系统、部分 DB 索引；节点多路，降低树高，适配磁盘页。
本实现为内存版 t-阶 B 树（最少 t-1，最多 2t-1 keys）。
"""


class BTreeNode:
    def __init__(self, t, leaf=True):
        self.t = t
        self.leaf = leaf
        self.keys = []
        self.vals = []
        self.children = []


class BTree:
    def __init__(self, t=3):
        self.t = t
        self.root = BTreeNode(t)

    # ---------- query ----------
    def query(self, key):
        return self._query(self.root, key)

    def _query(self, node, key):
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        if i < len(node.keys) and key == node.keys[i]:
            return node.vals[i]
        if node.leaf:
            return None
        return self._query(node.children[i], key)

    # ---------- update ----------
    def update(self, key, val):
        node, idx = self._find(self.root, key)
        if node is None:
            return False
        node.vals[idx] = val
        return True

    def _find(self, node, key):
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        if i < len(node.keys) and key == node.keys[i]:
            return node, i
        if node.leaf:
            return None, -1
        return self._find(node.children[i], key)

    # ---------- add ----------
    def add(self, key, val=None):
        val = key if val is None else val
        if self.query(key) is not None:
            self.update(key, val)
            return
        r = self.root
        if len(r.keys) == 2 * self.t - 1:
            s = BTreeNode(self.t, leaf=False)
            s.children = [r]
            self.root = s
            self._split_child(s, 0)
            self._insert_nonfull(s, key, val)
        else:
            self._insert_nonfull(r, key, val)

    def _split_child(self, parent, i):
        t = self.t
        y = parent.children[i]
        z = BTreeNode(t, leaf=y.leaf)
        mid = t - 1
        z.keys = y.keys[t:]
        z.vals = y.vals[t:]
        if not y.leaf:
            z.children = y.children[t:]
            y.children = y.children[:t]
        parent.keys.insert(i, y.keys[mid])
        parent.vals.insert(i, y.vals[mid])
        parent.children.insert(i + 1, z)
        y.keys = y.keys[:mid]
        y.vals = y.vals[:mid]

    def _insert_nonfull(self, node, key, val):
        i = len(node.keys) - 1
        if node.leaf:
            node.keys.append(None)
            node.vals.append(None)
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                node.vals[i + 1] = node.vals[i]
                i -= 1
            node.keys[i + 1] = key
            node.vals[i + 1] = val
        else:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            if len(node.children[i].keys) == 2 * self.t - 1:
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            self._insert_nonfull(node.children[i], key, val)

    # ---------- del ----------
    def delete(self, key):
        self._delete(self.root, key)
        if not self.root.keys and self.root.children:
            self.root = self.root.children[0]

    def _delete(self, node, key):
        t = self.t
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        if node.leaf:
            if i < len(node.keys) and node.keys[i] == key:
                node.keys.pop(i)
                node.vals.pop(i)
            return
        if i < len(node.keys) and node.keys[i] == key:
            if len(node.children[i].keys) >= t:
                pred_k, pred_v = self._max_kv(node.children[i])
                node.keys[i], node.vals[i] = pred_k, pred_v
                self._delete(node.children[i], pred_k)
            elif len(node.children[i + 1].keys) >= t:
                succ_k, succ_v = self._min_kv(node.children[i + 1])
                node.keys[i], node.vals[i] = succ_k, succ_v
                self._delete(node.children[i + 1], succ_k)
            else:
                self._merge(node, i)
                self._delete(node.children[i], key)
        else:
            if len(node.children[i].keys) < t:
                self._fill(node, i)
                if i > len(node.keys):
                    i -= 1
            self._delete(node.children[i], key)

    def _min_kv(self, node):
        while not node.leaf:
            node = node.children[0]
        return node.keys[0], node.vals[0]

    def _max_kv(self, node):
        while not node.leaf:
            node = node.children[-1]
        return node.keys[-1], node.vals[-1]

    def _merge(self, node, i):
        c = node.children[i]
        s = node.children[i + 1]
        c.keys.append(node.keys[i])
        c.vals.append(node.vals[i])
        c.keys += s.keys
        c.vals += s.vals
        if not c.leaf:
            c.children += s.children
        node.keys.pop(i)
        node.vals.pop(i)
        node.children.pop(i + 1)

    def _fill(self, node, i):
        t = self.t
        if i > 0 and len(node.children[i - 1].keys) >= t:
            self._borrow_prev(node, i)
        elif i < len(node.keys) and len(node.children[i + 1].keys) >= t:
            self._borrow_next(node, i)
        else:
            if i < len(node.keys):
                self._merge(node, i)
            else:
                self._merge(node, i - 1)

    def _borrow_prev(self, node, i):
        c, s = node.children[i], node.children[i - 1]
        c.keys.insert(0, node.keys[i - 1])
        c.vals.insert(0, node.vals[i - 1])
        if not c.leaf:
            c.children.insert(0, s.children.pop())
        node.keys[i - 1] = s.keys.pop()
        node.vals[i - 1] = s.vals.pop()

    def _borrow_next(self, node, i):
        c, s = node.children[i], node.children[i + 1]
        c.keys.append(node.keys[i])
        c.vals.append(node.vals[i])
        if not c.leaf:
            c.children.append(s.children.pop(0))
        node.keys[i] = s.keys.pop(0)
        node.vals[i] = s.vals.pop(0)

    def keys_inorder(self):
        out = []

        def dfs(n):
            if n.leaf:
                out.extend(n.keys)
                return
            for i, k in enumerate(n.keys):
                dfs(n.children[i])
                out.append(k)
            dfs(n.children[-1])

        dfs(self.root)
        return out


def _test():
    t = BTree(t=3)
    for k in [10, 20, 5, 6, 12, 30, 7, 17]:
        t.add(k, k * 10)
    assert t.query(6) == 60
    assert t.update(6, 66) and t.query(6) == 66
    assert t.keys_inorder() == [5, 6, 7, 10, 12, 17, 20, 30]
    t.delete(6)
    assert t.query(6) is None
    t.delete(13)  # no-op missing
    assert 6 not in t.keys_inorder()
    print("BTree OK", t.keys_inorder())


APPLICATION = """
- 经典磁盘索引：一次 I/O 读一整页多 key，树高低。
- MongoDB WiredTiger / 许多文件系统目录索引用 B/B* 变体。
- vs B+: B 树内部节点也存 value；B+ 数据只在叶子 + 叶子链表，范围扫更优。
- vs LSM: B 树就地更新 → 随机写多；LSM 顺序写更适合写密集。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
