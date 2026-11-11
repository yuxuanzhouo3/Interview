"""
7. KD-Tree (k-dimensional BST)
Template: query(nearest) / update / add / del
App: 空间划分、KNN、射线/范围查询；游戏、GIS、推荐近邻（低维）。
"""

import math


class KDNode:
    def __init__(self, point, axis, val=None):
        self.point = point  # tuple
        self.val = val if val is not None else point
        self.axis = axis
        self.left = None
        self.right = None


class KDTree:
    def __init__(self, k=2):
        self.k = k
        self.root = None

    # ---------- add ----------
    def add(self, point, val=None):
        point = tuple(point)
        self.root = self._insert(self.root, point, val if val is not None else point, 0)

    def _insert(self, node, point, val, depth):
        if not node:
            return KDNode(point, depth % self.k, val)
        axis = node.axis
        if point[axis] < node.point[axis]:
            node.left = self._insert(node.left, point, val, depth + 1)
        else:
            # equal goes right; exact same point → update
            if point == node.point:
                node.val = val
                return node
            node.right = self._insert(node.right, point, val, depth + 1)
        return node

    # ---------- query: exact ----------
    def query(self, point):
        point = tuple(point)
        cur = self.root
        depth = 0
        while cur:
            if cur.point == point:
                return cur.val
            axis = depth % self.k
            cur = cur.left if point[axis] < cur.point[axis] else cur.right
            depth += 1
        return None

    # ---------- query: nearest ----------
    def query_nearest(self, point):
        point = tuple(point)
        best = [None, float("inf")]  # node, dist2

        def dist2(a, b):
            return sum((x - y) ** 2 for x, y in zip(a, b))

        def dfs(node, depth):
            if not node:
                return
            d = dist2(point, node.point)
            if d < best[1]:
                best[0], best[1] = node, d
            axis = depth % self.k
            diff = point[axis] - node.point[axis]
            first, second = (node.left, node.right) if diff < 0 else (node.right, node.left)
            dfs(first, depth + 1)
            if diff * diff < best[1]:
                dfs(second, depth + 1)

        dfs(self.root, 0)
        return None if not best[0] else (best[0].point, best[0].val, math.sqrt(best[1]))

    # ---------- update ----------
    def update(self, point, val):
        point = tuple(point)
        cur = self.root
        depth = 0
        while cur:
            if cur.point == point:
                cur.val = val
                return True
            axis = depth % self.k
            cur = cur.left if point[axis] < cur.point[axis] else cur.right
            depth += 1
        return False

    # ---------- del ----------
    def delete(self, point):
        point = tuple(point)
        self.root = self._delete(self.root, point, 0)

    def _delete(self, node, point, depth):
        if not node:
            return None
        axis = depth % self.k
        if node.point == point:
            # 简化: 收集子树重建
            pts = []
            self._collect(node.left, pts)
            self._collect(node.right, pts)
            if not pts:
                return None
            new_root = None
            for p, v in pts:
                new_root = self._insert(new_root, p, v, depth)
            return new_root
        if point[axis] < node.point[axis]:
            node.left = self._delete(node.left, point, depth + 1)
        else:
            node.right = self._delete(node.right, point, depth + 1)
        return node

    def _collect(self, node, out):
        if not node:
            return
        out.append((node.point, node.val))
        self._collect(node.left, out)
        self._collect(node.right, out)


def _test():
    t = KDTree(2)
    for p in [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]:
        t.add(p, p)
    assert t.query((5, 4)) == (5, 4)
    assert t.update((5, 4), "P") and t.query((5, 4)) == "P"
    nn = t.query_nearest((9, 2))
    assert nn[0] in [(8, 1), (7, 2), (9, 6)]
    t.delete((5, 4))
    assert t.query((5, 4)) is None
    print("KD OK nearest", nn)


APPLICATION = """
- 低维空间近邻/范围查询（k 小，如 2D/3D）；高维失效（维度灾难）→ LSH / 向量索引。
- GIS、碰撞检测、机器人路径、简单推荐近邻。
- 对比: Quad/Octree 规则空间剖分；R-Tree 更适合矩形范围（DB 空间索引）。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
