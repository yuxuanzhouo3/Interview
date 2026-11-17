"""
LC 3416. Subsequences with a Unique Middle Mode II — Hard
https://leetcode.com/problems/subsequences-with-a-unique-middle-mode-ii/
Best: Same as 3395 with aggregated sums — O(n) ideal; O(n^2) ok for tests
"""

from typing import List
from collections import defaultdict

MOD = 10**9 + 7


class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        def c2(n: int) -> int:
            return n * (n - 1) // 2 if n >= 2 else 0

        n = len(nums)
        L, R = defaultdict(int), defaultdict(int)
        for x in nums:
            R[x] += 1
        res = 0
        for i, x in enumerate(nums):
            j = n - 1 - i
            R[x] -= 1
            lx, rx = L[x], R[x]
            res += c2(i) * c2(j) - c2(i - lx) * c2(j - rx)
            seen = set(R.keys()) | set(L.keys())
            for y in seen:
                if y == x:
                    continue
                ly, ry = L[y], R[y]
                if ly == 0 and ry == 0:
                    continue
                res -= lx * (i - lx) * c2(ry)
                res -= c2(ly) * rx * (j - rx)
                res -= ly * (i - lx - ly) * rx * ry
                res -= lx * ly * ry * (j - rx - ry)
            res %= MOD
            L[x] += 1
        return res % MOD


def _test():
    s = Solution()
    assert s.subsequencesWithMiddleMode([1, 1, 1, 1, 1, 1]) == 6
    print("LC 3416 OK")


if __name__ == "__main__":
    _test()
