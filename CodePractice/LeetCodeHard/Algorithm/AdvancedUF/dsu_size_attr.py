"""
2. DSU 维护 size / 边数等属性
"""


class DSUAttr:
    def __init__(self, n):
        self.p = list(range(n))
        self.size = [1] * n
        self.edges = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            self.edges[ra] += 1
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.size[ra] += self.size[rb]
        self.edges[ra] += self.edges[rb] + 1
        return True

    def query(self, x, attr="size"):
        r = self.find(x)
        return self.size[r] if attr == "size" else self.edges[r]

    def update(self, a, b):
        return self.union(a, b)

    def add(self):
        self.p.append(len(self.p))
        self.size.append(1)
        self.edges.append(0)
        return len(self.p) - 1

    def delete(self, x):
        # 教学：仅重置自身，不拆分量
        pass


def _test():
    d = DSUAttr(4)
    d.union(0, 1)
    d.union(1, 2)
    assert d.query(0) == 3
    d.union(0, 2)  # same component, edges++
    assert d.query(0, "edges") == 3
    print("dsu_size_attr OK", d.query(2), d.query(0, "edges"))


APPLICATION = """
- 合并时属性相加；同根再连边只加 edges。
- LC 827 最大岛周围扩展要用 size。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
