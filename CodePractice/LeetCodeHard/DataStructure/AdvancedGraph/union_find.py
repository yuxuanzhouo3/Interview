"""
AdvancedGraph / Union-Find (DSU)
Template: query(find/connected) / update(union) / add(node) / del(rare)
App: 连通分量、Kruskal、账号合并、网格岛问题。
"""


class UnionFind:
    def __init__(self, n=0):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n

    def add(self, x=None):
        """扩展一个新节点；或确保容量。"""
        if x is None:
            x = len(self.parent)
        while len(self.parent) <= x:
            self.parent.append(len(self.parent))
            self.rank.append(0)
            self.count += 1
        return x

    def query(self, x):
        """find 根。"""
        return self.find(x)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def connected(self, a, b):
        return self.find(a) == self.find(b)

    def update(self, a, b):
        """union；返回是否合并成功。"""
        return self.union(a, b)

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.count -= 1
        return True

    def delete(self, x):
        """
        并查集一般不支持高效删点；教学版：重置为独立集合
        （不修复历史上挂到它下面的关系，仅示意）。
        """
        self.parent[x] = x
        self.rank[x] = 0


def _test():
    uf = UnionFind(5)
    assert uf.update(0, 1)
    assert uf.update(1, 2)
    assert uf.connected(0, 2)
    assert uf.query(0) == uf.query(2)
    uf.add()  # node 5
    assert not uf.connected(5, 0)
    uf.update(5, 0)
    assert uf.connected(5, 2)
    print("UnionFind OK components~", uf.count)


APPLICATION = """
- Kruskal MST、朋友圈、网格染色等价类、账户合并 LC 721。
- 分布式: 某些一致性哈希迁移用连通思想；生产图存储仍邻接表+外部索引。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
