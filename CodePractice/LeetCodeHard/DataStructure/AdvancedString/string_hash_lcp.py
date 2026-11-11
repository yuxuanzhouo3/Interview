"""
8. Double String Hash + LCP via hash
Template: query(hash/lcp/equal) / update(point rebuild) / add(append) / del(pop)
App: 快速判相等、二分 LCP、子串字典序比较；竞赛替代 SA 部分场景。
"""


class DoubleHash:
    def __init__(self, s="", mod1=10**9 + 7, mod2=10**9 + 9, base=911382323):
        self.mod1, self.mod2, self.base = mod1, mod2, base
        self.s = []
        self.p1, self.p2 = [0], [0]
        self.pow1, self.pow2 = [1], [1]
        if s:
            self.add(s)

    def add(self, chars):
        for ch in chars:
            o = ord(ch)
            self.s.append(ch)
            self.pow1.append(self.pow1[-1] * self.base % self.mod1)
            self.pow2.append(self.pow2[-1] * self.base % self.mod2)
            self.p1.append((self.p1[-1] * self.base + o) % self.mod1)
            self.p2.append((self.p2[-1] * self.base + o) % self.mod2)

    def query(self, l, r=None):
        """hash of s[l:r]"""
        if r is None:
            r = len(self.s)
        x = (self.p1[r] - self.p1[l] * self.pow1[r - l]) % self.mod1
        y = (self.p2[r] - self.p2[l] * self.pow2[r - l]) % self.mod2
        return (x, y)

    def equal(self, l1, r1, l2, r2):
        return self.query(l1, r1) == self.query(l2, r2)

    def lcp(self, i, j):
        n = len(self.s)
        lo, hi = 0, n - max(i, j)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.equal(i, i + mid, j, j + mid):
                lo = mid
            else:
                hi = mid - 1
        return lo

    def update(self, i, ch):
        if not (0 <= i < len(self.s)):
            return False
        body = list(self.s)
        body[i] = ch
        mods = (self.mod1, self.mod2, self.base)
        self.mod1, self.mod2, self.base = mods
        self.s, self.p1, self.p2 = [], [0], [0]
        self.pow1, self.pow2 = [1], [1]
        self.add("".join(body))
        return True

    def delete(self, n=1):
        n = min(n, len(self.s))
        for _ in range(n):
            self.s.pop()
            self.p1.pop()
            self.p2.pop()
            self.pow1.pop()
            self.pow2.pop()


def _test():
    h = DoubleHash("ababab")
    assert h.equal(0, 2, 2, 4)
    assert h.lcp(0, 2) == 4
    assert h.update(0, "x")
    assert not h.equal(0, 2, 2, 4)
    h.delete(2)
    assert len(h.s) == 4
    print("DoubleHash OK LCP", DoubleHash("ababab").lcp(0, 2))


APPLICATION = """
- 双模降冲突；子串比较/LCP 二分 O(log n) 每询。
- 工程去重指纹、CDC；竞赛比 SA 好写。
- Java: 自写 long 双模；注意溢出用 Math.floorMod。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
