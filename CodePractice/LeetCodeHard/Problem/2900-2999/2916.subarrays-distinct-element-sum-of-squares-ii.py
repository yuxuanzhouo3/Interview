"""
LC 2916. Subarrays Distinct Element Sum of Squares II — Hard
Best: Segment tree last occurrence contribution — O(n log n)
"""

from typing import List


class SegTree:
    def __init__(self, n):
        self.n = n
        self.t = [0] * (4 * n)
        self.lazy = [0] * (4 * n)

    def _push(self, p, l, r):
        if self.lazy[p] and l != r:
            mid = (l + r) // 2
            for c, lc, rc in ((2 * p, l, mid), (2 * p + 1, mid + 1, r)):
                self.t[c] += self.lazy[p] * (rc - lc + 1)
                self.lazy[c] += self.lazy[p]
            self.lazy[p] = 0

    def add(self, p, l, r, ql, qr, v):
        if qr < l or r < ql:
            return
        if ql <= l and r <= qr:
            self.t[p] += v * (r - l + 1)
            self.lazy[p] += v
            return
        self._push(p, l, r)
        mid = (l + r) // 2
        self.add(2 * p, l, mid, ql, qr, v)
        self.add(2 * p + 1, mid + 1, r, ql, qr, v)
        self.t[p] = self.t[2 * p] + self.t[2 * p + 1]

    def query(self, p, l, r, ql, qr):
        if qr < l or r < ql:
            return 0
        if ql <= l and r <= qr:
            return self.t[p]
        self._push(p, l, r)
        mid = (l + r) // 2
        return self.query(2 * p, l, mid, ql, qr) + self.query(2 * p + 1, mid + 1, r, ql, qr)


class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        last = {}
        # Maintain distinct count array d[j] for subarrays ending at i starting at j
        # sum of squares via: when extending, update range (prev+1..i)
        st = SegTree(n)
        ans = cur = 0
        for i, x in enumerate(nums):
            prev = last.get(x, -1)
            # add 1 to distinct for starts in (prev, i]
            # contribution to sum of squares: 2*old+1 for each
            # use: query sum of distinct on [prev+1,i], then add length
            if prev + 1 <= i:
                s = st.query(1, 0, n - 1, prev + 1, i)
                cur = (cur + 2 * s + (i - prev)) % MOD
                st.add(1, 0, n - 1, prev + 1, i, 1)
            last[x] = i
            ans = (ans + cur) % MOD
        return ans


def _test():
    s = Solution()
    assert s.sumCounts([1, 2, 1]) == 15
    print("LC 2916 OK")


if __name__ == "__main__":
    _test()
