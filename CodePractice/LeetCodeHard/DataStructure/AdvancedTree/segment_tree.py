"""
9. Segment Tree
Template: query(range) / update(point|range+lazy) / add / del
App: 区间和/最值、扫描线、日程；LeetCode Hard 常客。
"""


class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        if arr:
            self._build(1, 0, self.n - 1, arr)

    def _build(self, p, l, r, arr):
        if l == r:
            self.tree[p] = arr[l]
            return
        m = (l + r) // 2
        self._build(p * 2, l, m, arr)
        self._build(p * 2 + 1, m + 1, r, arr)
        self.tree[p] = self.tree[p * 2] + self.tree[p * 2 + 1]

    def _push(self, p, l, r):
        if self.lazy[p] == 0 or l == r:
            return
        mid = (l + r) // 2
        for child, cl, cr in ((p * 2, l, mid), (p * 2 + 1, mid + 1, r)):
            self.tree[child] += self.lazy[p] * (cr - cl + 1)
            self.lazy[child] += self.lazy[p]
        self.lazy[p] = 0

    # ---------- query ----------
    def query(self, ql, qr):
        return self._query(1, 0, self.n - 1, ql, qr)

    def _query(self, p, l, r, ql, qr):
        if qr < l or r < ql:
            return 0
        if ql <= l and r <= qr:
            return self.tree[p]
        self._push(p, l, r)
        m = (l + r) // 2
        return self._query(p * 2, l, m, ql, qr) + self._query(p * 2 + 1, m + 1, r, ql, qr)

    # ---------- update point ----------
    def update(self, idx, val):
        """单点设值。"""
        self._update_point(1, 0, self.n - 1, idx, val)

    def _update_point(self, p, l, r, idx, val):
        if l == r:
            self.tree[p] = val
            return
        self._push(p, l, r)
        m = (l + r) // 2
        if idx <= m:
            self._update_point(p * 2, l, m, idx, val)
        else:
            self._update_point(p * 2 + 1, m + 1, r, idx, val)
        self.tree[p] = self.tree[p * 2] + self.tree[p * 2 + 1]

    # ---------- add range ----------
    def add(self, ql, qr, delta):
        """区间加（懒标记）。单点 add → ql==qr。"""
        self._add(1, 0, self.n - 1, ql, qr, delta)

    def _add(self, p, l, r, ql, qr, delta):
        if qr < l or r < ql:
            return
        if ql <= l and r <= qr:
            self.tree[p] += delta * (r - l + 1)
            self.lazy[p] += delta
            return
        self._push(p, l, r)
        m = (l + r) // 2
        self._add(p * 2, l, m, ql, qr, delta)
        self._add(p * 2 + 1, m + 1, r, ql, qr, delta)
        self.tree[p] = self.tree[p * 2] + self.tree[p * 2 + 1]

    # ---------- del ----------
    def delete(self, idx):
        """逻辑删除：置 0。"""
        self.update(idx, 0)


def _test():
    st = SegmentTree([1, 2, 3, 4, 5])
    assert st.query(0, 4) == 15
    assert st.query(1, 3) == 9
    st.update(2, 10)
    assert st.query(0, 4) == 22
    st.add(0, 2, 1)  # +1 on [0,2]
    assert st.query(0, 4) == 25
    st.delete(4)
    assert st.query(0, 4) == 20
    print("SegmentTree OK", st.query(0, 4))


APPLICATION = """
- 竞赛/面试: 区间更新+查询；扫描线矩形面积并。
- 工业: 广告库存区间、日历忙闲、游戏伤害区间 buff。
- vs BIT: 线段树更通用（min/max/gcd/懒标记）；BIT 更轻。
- LeetCode: 307, 315, 218 The Skyline Problem, 327 Count of Range Sum.
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
