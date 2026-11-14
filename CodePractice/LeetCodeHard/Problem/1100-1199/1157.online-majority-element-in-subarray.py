"""
LC 1157. Online Majority Element In Subarray — Hard
https://leetcode.com/problems/online-majority-element-in-subarray/
Best: Segment tree majority candidate + binary search count — O(n log n + q log^2 n)
"""

from typing import List
from collections import defaultdict
import bisect


class MajorityChecker:
    def __init__(self, arr: List[int]):
        self.arr = arr
        self.n = len(arr)
        self.pos = defaultdict(list)
        for i, v in enumerate(arr):
            self.pos[v].append(i)
        self.tree = [0] * (4 * self.n)
        self._build(1, 0, self.n - 1)

    def _count(self, val, left, right):
        a = self.pos[val]
        return bisect.bisect_right(a, right) - bisect.bisect_left(a, left)

    def _merge(self, a, b, L, R):
        if a == 0:
            return b
        if b == 0:
            return a
        if a == b:
            return a
        # Boyer-Moore style on range via counts
        ca = self._count(a, L, R)
        cb = self._count(b, L, R)
        return a if ca >= cb else b

    def _build(self, idx, L, R):
        if L == R:
            self.tree[idx] = self.arr[L]
            return
        mid = (L + R) // 2
        self._build(idx * 2, L, mid)
        self._build(idx * 2 + 1, mid + 1, R)
        self.tree[idx] = self._merge(self.tree[idx * 2], self.tree[idx * 2 + 1], L, R)

    def _query_cand(self, idx, L, R, ql, qr):
        if qr < L or R < ql:
            return 0
        if ql <= L and R <= qr:
            return self.tree[idx]
        mid = (L + R) // 2
        left = self._query_cand(idx * 2, L, mid, ql, qr)
        right = self._query_cand(idx * 2 + 1, mid + 1, R, ql, qr)
        return self._merge(left, right, ql, qr)

    def query(self, left: int, right: int, threshold: int) -> int:
        cand = self._query_cand(1, 0, self.n - 1, left, right)
        if cand and self._count(cand, left, right) >= threshold:
            return cand
        return -1


def _test():
    m = MajorityChecker([1, 1, 2, 2, 1, 1])
    assert m.query(0, 5, 4) == 1
    assert m.query(0, 3, 3) == -1
    assert m.query(2, 3, 2) == 2
    print("LC 1157 OK")


if __name__ == "__main__":
    _test()
