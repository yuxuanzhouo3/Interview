"""
LC 2143. Choose Numbers From Two Arrays in Range — Hard
https://leetcode.com/problems/choose-numbers-from-two-arrays-in-range/
Best: 差分 DP：选 a 加、选 b 减，累计差为 0 — O(n * Σ)
"""

from typing import List
from collections import Counter


class Solution:
    def countSubranges(self, nums1: List[int], nums2: List[int]) -> int:
        MOD = 10**9 + 7
        ans = 0
        dp = Counter()
        for a, b in zip(nums1, nums2):
            ndp = Counter()
            ndp[a] += 1
            ndp[-b] += 1
            for s, c in dp.items():
                ndp[s + a] = (ndp[s + a] + c) % MOD
                ndp[s - b] = (ndp[s - b] + c) % MOD
            dp = ndp
            ans = (ans + dp[0]) % MOD
        return ans


def _test():
    s = Solution()
    assert s.countSubranges([1, 2, 5], [2, 6, 3]) == 3
    assert s.countSubranges([0, 1], [1, 0]) == 4
    print("LC 2143 OK")


if __name__ == "__main__":
    _test()
