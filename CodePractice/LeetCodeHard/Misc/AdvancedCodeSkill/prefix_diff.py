"""
CodeSkill / 前缀和 + 差分
Template: query(range_sum|array) / update(range) / add / del
App: 静态区间和；多次区间加。
"""


class PrefixDiff:
    def __init__(self, arr=None):
        self.a = list(arr or [])
        self._rebuild_prefix()
        self.diff = None

    def _rebuild_prefix(self):
        self.pref = [0]
        for x in self.a:
            self.pref.append(self.pref[-1] + x)

    def add(self, x):
        self.a.append(x)
        self.pref.append(self.pref[-1] + x)
        self.diff = None

    def update(self, l, r=None, val=None):
        """update(i, val) 单点；update(l,r,val) 差分区间加。"""
        if r is None or (val is None and not isinstance(r, int)):
            return False
        if val is None:
            # single point update(i, new_val) — here r is new val
            i, new_val = l, r
            if not (0 <= i < len(self.a)):
                return False
            self.a[i] = new_val
            self._rebuild_prefix()
            self.diff = None
            return True
        # range add via diff
        n = len(self.a)
        if self.diff is None:
            self.diff = [0] * (n + 1)
            self.diff[0] = self.a[0] if n else 0
            for i in range(1, n):
                self.diff[i] = self.a[i] - self.a[i - 1]
        self.diff[l] += val
        if r + 1 < n:
            self.diff[r + 1] -= val
        elif r + 1 == n:
            pass
        else:
            return False
        # materialize
        cur = 0
        for i in range(n):
            cur += self.diff[i]
            self.a[i] = cur
        self._rebuild_prefix()
        return True

    def delete(self, idx):
        if not (0 <= idx < len(self.a)):
            return False
        self.a.pop(idx)
        self._rebuild_prefix()
        self.diff = None
        return True

    def query(self, l=None, r=None):
        if l is None:
            return list(self.a)
        return self.pref[r + 1] - self.pref[l]


def _test():
    p = PrefixDiff([1, 2, 3, 4])
    assert p.query(1, 3) == 9
    assert p.update(0, 2, 5)  # +5 on [0,2]
    assert p.query() == [6, 7, 8, 4]
    assert p.update(3, 10)
    p.add(1)
    assert p.delete(0)
    print("PrefixDiff OK", p.query())


APPLICATION = """
- 静态和：前缀 O(1)；动态区间加多用差分或树状数组/线段树。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
