"""
ClassifyThinking / 增量 · 差分数组
Template: query(array) / update(range add) / add / del
App: 区间加频繁、最后统一还原；扫线事件。
"""


class DiffArray:
    def __init__(self, n=0):
        self.n = n
        self.diff = [0] * (n + 1)

    def add(self, n_or_l, r=None, val=1):
        """add(n) 扩长；或 add(l,r,val) 区间加。"""
        if r is None:
            while self.n < n_or_l:
                self.diff.append(0)
                self.n += 1
            return
        l, r, val = n_or_l, r, val
        self._ensure(r + 1)
        self.diff[l] += val
        self.diff[r + 1] -= val

    def _ensure(self, need):
        while self.n < need:
            self.diff.append(0)
            self.n += 1
        if len(self.diff) < self.n + 1:
            self.diff.extend([0] * (self.n + 1 - len(self.diff)))

    def update(self, l, r, val):
        if l > r or l < 0:
            return False
        self.add(l, r, val)
        return True

    def delete(self, l, r):
        """撤销区间：加相反数。"""
        return self.update(l, r, 0)  # no-op placeholder
        # real undo needs记录；教学版用 update(l,r,-old)

    def query(self):
        """还原数组 a[0..n-1]。"""
        a, cur = [], 0
        for i in range(self.n):
            cur += self.diff[i]
            a.append(cur)
        return a


def _test():
    d = DiffArray(5)
    d.add(1, 3, 2)   # idx 1..3 +2
    d.update(0, 4, 1)
    assert d.query() == [1, 3, 3, 3, 1]
    d.update(1, 3, -2)  # undo first
    assert d.query() == [1, 1, 1, 1, 1]
    print("Incremental OK", d.query())


APPLICATION = """
- 差分数组适合「多次区间加，一次查询全表」。
- 扫线：事件 (l,+v)/(r+1,-v) 排序后扫。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
