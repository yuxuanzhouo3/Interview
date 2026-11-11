"""
3. Red-Black Tree (approx-balanced BST)
Template: query / update / add / del
App: Java TreeMap/TreeSet, C++ std::map/set, Linux CFS 调度; 写比 AVL 便宜。
简化实现：用 BST + size 说明接口；完整 RB 旋转/染色见注释路线。
生产级可用 sortedcontainers 或标准库；本文件给可运行 CRUD + 红黑不变量说明。
"""


class Node:
    RED, BLACK = 0, 1

    def __init__(self, key, val=None, color=0):
        self.key = key
        self.val = val if val is not None else key
        self.color = color  # 0 red, 1 black
        self.left = None
        self.right = None
        self.parent = None


class RedBlackTree:
    """
    教学版：保持有序 BST 语义 + 颜色字段；
    insert 走标准 RB 修复骨架（左旋/右旋/变色）。delete 完整修复较繁，提供 BST 删除 + 说明。
    """

    def __init__(self):
        self.root = None

    def _rot_l(self, x):
        y = x.right
        x.right = y.left
        if y.left:
            y.left.parent = x
        y.parent = x.parent
        if not x.parent:
            self.root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _rot_r(self, y):
        x = y.left
        y.left = x.right
        if x.right:
            x.right.parent = y
        x.parent = y.parent
        if not y.parent:
            self.root = x
        elif y is y.parent.left:
            y.parent.left = x
        else:
            y.parent.right = x
        x.right = y
        y.parent = x

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
        val = key if val is None else val
        if not self.root:
            self.root = Node(key, val, Node.BLACK)
            return
        cur, parent = self.root, None
        while cur:
            parent = cur
            if key == cur.key:
                cur.val = val
                return
            cur = cur.left if key < cur.key else cur.right
        node = Node(key, val, Node.RED)
        node.parent = parent
        if key < parent.key:
            parent.left = node
        else:
            parent.right = node
        self._fix_insert(node)

    def _fix_insert(self, z):
        while z.parent and z.parent.color == Node.RED:
            p = z.parent
            g = p.parent
            if not g:
                break
            if p is g.left:
                u = g.right
                if u and u.color == Node.RED:
                    p.color = u.color = Node.BLACK
                    g.color = Node.RED
                    z = g
                else:
                    if z is p.right:
                        z = p
                        self._rot_l(z)
                        p = z.parent
                    p.color = Node.BLACK
                    g.color = Node.RED
                    self._rot_r(g)
            else:
                u = g.left
                if u and u.color == Node.RED:
                    p.color = u.color = Node.BLACK
                    g.color = Node.RED
                    z = g
                else:
                    if z is p.left:
                        z = p
                        self._rot_r(z)
                        p = z.parent
                    p.color = Node.BLACK
                    g.color = Node.RED
                    self._rot_l(g)
        self.root.color = Node.BLACK

    # ---------- del (BST delete; full RB delete-fixomitted for brevity) ----------
    def delete(self, key):
        self.root = self._bst_delete(self.root, key)
        if self.root:
            self.root.parent = None
            self.root.color = Node.BLACK

    def _bst_delete(self, node, key):
        if not node:
            return None
        if key < node.key:
            node.left = self._bst_delete(node.left, key)
            if node.left:
                node.left.parent = node
        elif key > node.key:
            node.right = self._bst_delete(node.right, key)
            if node.right:
                node.right.parent = node
        else:
            if not node.left:
                return node.right
            if not node.right:
                return node.left
            succ = node.right
            while succ.left:
                succ = succ.left
            node.key, node.val = succ.key, succ.val
            node.right = self._bst_delete(node.right, succ.key)
            if node.right:
                node.right.parent = node
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
    t = RedBlackTree()
    for k in [10, 20, 30, 15, 25, 5, 1]:
        t.add(k, k)
    assert t.inorder() == [1, 5, 10, 15, 20, 25, 30]
    assert t.query(15) == 15
    assert t.update(15, 150) and t.query(15) == 150
    t.delete(20)
    assert t.query(20) is None
    assert 20 not in t.inorder()
    print("RedBlack OK", t.inorder(), "root_color", t.root.color)


APPLICATION = """
- Java: TreeMap / TreeSet / ConcurrentSkipListMap(跳表) vs HashMap(无序 O(1))。
  Hashtable 是遗留同步哈希表；有序需求 → TreeMap（红黑）。
- C++: std::map / std::set；Linux 内核 CFS 用红黑树管 vruntime。
- 为何工程偏爱 RB 而非 AVL：插入最多 2 次旋转 + 染色，写路径更轻；
  查询仍 O(log n)，平衡稍松（黑高约束）。
- 磁盘场景仍不选 RB → 选 B/B+（扇区友好）或 LSM（写友好）。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
