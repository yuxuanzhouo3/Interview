"""
LC 1819. Number of Different Subsequences GCDs — Hard
https://leetcode.com/problems/number-of-different-subsequences-gcds/
Best: for each g, gcd multiples that appear — O(M log M)
"""

from typing import List
from math import gcd


class Solution:
    def countDifferentSubsequenceGCDs(self, nums: List[int]) -> int:
        mx = max(nums)
        has = [False] * (mx + 1)
        for x in nums:
            has[x] = True
        ans = 0
        for g in range(1, mx + 1):
            cur = 0
            for mult in range(g, mx + 1, g):
                if has[mult]:
                    cur = gcd(cur, mult)
                    if cur == g:
                        ans += 1
                        break
        return ans


def _test():
    s = Solution()
    assert s.countDifferentSubsequenceGCDs([6, 10, 3]) == 5
    assert s.countDifferentSubsequenceGCDs([5, 15, 40, 5, 6]) == 7
    print("LC 1819 OK")


if __name__ == "__main__":
    _test()
