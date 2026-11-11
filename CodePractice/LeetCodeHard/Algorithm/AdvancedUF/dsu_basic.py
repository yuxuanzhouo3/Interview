"""
1. DSU 基础：路径压缩 + 按秩
Template: query(find/connected/count) / update(union) / add / del(reset)
"""


class DSU:
    def __init__(self, n=0):
        self.p = list(range(n))
        self.r = [0] * n
        self.count = n

    def add(self):
        self.p.append(len(self.p))
        self.r.append(0)
        self.count += 1
        return len(self.p) - 1

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def query(self, a, b=None):
        if b is None:
            return self.find(a)
        return self.find(a) == self.find(b)

    def update(self, a, b):
        return self.union(a, b)

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        self.count -= 1
        return True

    def delete(self, x):
        self.p[x] = x
        self.r[x] = 0


def _test():
    d = DSU(5)
    assert d.update(0, 1) and d.update(1, 2)
    assert d.query(0, 2) and d.count == 3
    d.add()
    assert d.count == 4
    print("dsu_basic OK", d.count)


APPLICATION = """
- 连通分量基线；Kruskal / 岛屿 / 账号合并。
- 均摊几乎 O(1)。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
