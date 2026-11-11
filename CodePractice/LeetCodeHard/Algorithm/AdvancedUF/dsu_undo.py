"""
6. 可撤销并查集（不路径压缩，按秩 + 栈回滚）
App: 分治回滚、动态图连通。
"""


class UndoDSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n
        self.stk = []  # (rb, ra, rank_changed)
        self.count = n

    def find(self, x):
        while self.p[x] != x:
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            self.stk.append(None)
            return False
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        changed = self.r[ra] == self.r[rb]
        if changed:
            self.r[ra] += 1
        self.stk.append((rb, ra, changed))
        self.count -= 1
        return True

    def undo(self):
        if not self.stk:
            return False
        rec = self.stk.pop()
        if rec is None:
            return True
        rb, ra, changed = rec
        self.p[rb] = rb
        if changed:
            self.r[ra] -= 1
        self.count += 1
        return True

    def query(self, a, b=None):
        if b is None:
            return self.find(a)
        return self.find(a) == self.find(b)

    def update(self, a, b):
        return self.union(a, b)

    def add(self):
        self.p.append(len(self.p))
        self.r.append(0)
        self.count += 1
        return len(self.p) - 1

    def delete(self):
        return self.undo()


def _test():
    d = UndoDSU(4)
    d.union(0, 1)
    d.union(2, 3)
    assert d.count == 2
    d.union(1, 2)
    assert d.query(0, 3)
    d.undo()
    assert not d.query(0, 3) and d.query(0, 1)
    d.undo()
    assert not d.query(2, 3)
    print("dsu_undo OK", d.count)


APPLICATION = """
- 禁止路径压缩；每次 union 记修改，undo 弹出。
- 时间线分治：处理中点询问，回滚右半边修改。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
