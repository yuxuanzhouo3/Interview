"""
LC 1649. Create Sorted Array through Instructions — Hard
https://leetcode.com/problems/create-sorted-array-through-instructions/
Best: Fenwick / BIT — O(n log M)
"""

from typing import List


class Fenwick:
    def __init__(self, n):
        self.n = n
        self.bit = [0] * (n + 1)

    def add(self, i, v=1):
        while i <= self.n:
            self.bit[i] += v
            i += i & -i

    def sum(self, i):
        s = 0
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return s


class Solution:
    def createSortedArray(self, instructions: List[int]) -> int:
        MOD = 10**9 + 7
        mx = max(instructions)
        fw = Fenwick(mx)
        ans = 0
        for i, x in enumerate(instructions):
            less = fw.sum(x - 1)
            greater = i - fw.sum(x)
            ans = (ans + min(less, greater)) % MOD
            fw.add(x)
        return ans


def _test():
    s = Solution()
    assert s.createSortedArray([1, 5, 6, 2]) == 1
    assert s.createSortedArray([1, 2, 3, 6, 5, 4]) == 3
    assert s.createSortedArray([1, 3, 3, 3, 2, 4, 2, 1, 2]) == 4
    print("LC 1649 OK")


if __name__ == "__main__":
    _test()
