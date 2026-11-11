"""
5. 带权并查集 / 扩展域
App: LC 990 等式方程；相对关系。
"""


class WeightedDSU:
    """delta[x] = val[x] - val[parent] 约定；find 后 val[x]-val[root]=delta[x]。"""

    def __init__(self, n):
        self.p = list(range(n))
        self.d = [0] * n  # to parent

    def find(self, x):
        if self.p[x] != x:
            fa = self.p[x]
            self.p[x] = self.find(fa)
            self.d[x] += self.d[fa]
        return self.p[x]

    def query(self, a, b):
        """若同根，返回 val[a]-val[b]；否则 None。"""
        if self.find(a) != self.find(b):
            return None
        return self.d[a] - self.d[b]

    def update(self, a, b, diff):
        """约束 val[a]-val[b]=diff；冲突返回 False。"""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return self.d[a] - self.d[b] == diff
        self.p[ra] = rb
        self.d[ra] = diff - self.d[a] + self.d[b]
        return True

    def add(self):
        self.p.append(len(self.p))
        self.d.append(0)
        return len(self.p) - 1

    def delete(self, x):
        return False


def equations_possible(equations):
    """LC 990: 'a==b' / 'a!=b'."""
    dsu = WeightedDSU(26)

    def idx(c):
        return ord(c) - 97

    for eq in equations:
        if eq[1] == "=":
            dsu.update(idx(eq[0]), idx(eq[3]), 0)
    for eq in equations:
        if eq[1] == "!":
            if dsu.find(idx(eq[0])) == dsu.find(idx(eq[3])) and dsu.query(idx(eq[0]), idx(eq[3])) == 0:
                return False
    return True


def _test():
    w = WeightedDSU(3)
    assert w.update(0, 1, 5)  # v0-v1=5
    assert w.update(1, 2, 3)  # v1-v2=3 → v0-v2=8
    assert w.query(0, 2) == 8
    assert equations_possible(["a==b", "b!=a"]) is False
    assert equations_possible(["a==b", "b==c", "a==c"]) is True
    print("dsu_weighted OK", w.query(0, 2))


APPLICATION = """
- find 时累加到根的权差；union 挂根并算新 delta。
- 扩展域：敌友拆成 x / x+n 两点。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
