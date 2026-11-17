"""
LC 3569. Maximize Count of Distinct Primes After Split — Hard
https://leetcode.com/problems/maximize-count-of-distinct-primes-after-split/
Best: Track prime endpoints + lazy segment tree — O((n+q) log n)
"""

from __future__ import annotations

import collections
from typing import List

from sortedcontainers import SortedList


def _linear_sieve(n: int) -> List[int]:
    primes: List[int] = []
    spf = [-1] * (n + 1)
    for i in range(2, n + 1):
        if spf[i] == -1:
            spf[i] = i
            primes.append(i)
        for p in primes:
            if i * p > n or p > spf[i]:
                break
            spf[i * p] = p
    return spf


MAX_N = 10**5
SPF = _linear_sieve(MAX_N)


class SegmentTree:
    def __init__(self, n: int):
        self.base = 1 << (n - 1).bit_length()
        self.tree = [0] * (self.base * 2)
        self.lazy = [0] * self.base

    def _apply(self, x: int, val: int) -> None:
        self.tree[x] += val
        if x < self.base:
            self.lazy[x] += val

    def _push(self, x: int) -> None:
        for h in range(x.bit_length() - 1, 0, -1):
            y = x >> h
            if self.lazy[y]:
                self._apply(y << 1, self.lazy[y])
                self._apply((y << 1) + 1, self.lazy[y])
                self.lazy[y] = 0

    def _pull(self, x: int) -> None:
        while x > 1:
            x >>= 1
            self.tree[x] = max(self.tree[x << 1], self.tree[(x << 1) + 1]) + self.lazy[x]

    def update(self, L: int, R: int, h: int) -> None:
        if L > R:
            return
        L += self.base
        R += self.base
        L0, R0 = L, R
        while L <= R:
            if L & 1:
                self._apply(L, h)
                L += 1
            if R & 1 == 0:
                self._apply(R, h)
                R -= 1
            L >>= 1
            R >>= 1
        self._pull(L0)
        self._pull(R0)


class Solution:
    def maximumCount(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        lookup: dict[int, SortedList] = collections.defaultdict(SortedList)
        st = SegmentTree(n - 1)

        def add(i: int, d: int) -> None:
            x = nums[i]
            if SPF[x] != x:
                return
            if d == 1:
                lookup[x].add(i)
            if len(lookup[x]) == 1:
                st.update(0, n - 2, d)
            elif i == lookup[x][0]:
                st.update(i, lookup[x][1] - 1, d)
            elif i == lookup[x][-1]:
                st.update(lookup[x][-2], i - 1, d)
            if d == -1:
                lookup[x].remove(i)

        for i in range(n):
            add(i, 1)

        ans = [0] * len(queries)
        for qi, (idx, x) in enumerate(queries):
            if nums[idx] != x:
                add(idx, -1)
                nums[idx] = x
                add(idx, 1)
            ans[qi] = st.tree[1]
        return ans


def _test():
    s = Solution()
    assert s.maximumCount([2, 1, 3, 1, 2], [[1, 2], [3, 3]]) == [3, 4]
    assert s.maximumCount([2, 1, 4], [[0, 1]]) == [0]
    print("LC 3569 OK")


if __name__ == "__main__":
    _test()
