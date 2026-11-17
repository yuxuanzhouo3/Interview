"""
LC 3686. Number of Stable Subsequences — Hard
https://leetcode.com/problems/number-of-stable-subsequences/
Best: DP by parity run length — O(n)
"""

from typing import List


class Solution:
    def countStableSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        a1 = a2 = b1 = b2 = 0
        for x in nums:
            if x & 1:
                na1 = (1 + b1 + b2) % MOD
                na2 = a1
                a1 = (a1 + na1) % MOD
                a2 = (a2 + na2) % MOD
            else:
                nb1 = (1 + a1 + a2) % MOD
                nb2 = b1
                b1 = (b1 + nb1) % MOD
                b2 = (b2 + nb2) % MOD
        return (a1 + a2 + b1 + b2) % MOD


def _test():
    s = Solution()
    assert s.countStableSubsequences([1, 3, 5]) == 6
    assert s.countStableSubsequences([2, 3, 4, 2]) == 14
    print("LC 3686 OK")


if __name__ == "__main__":
    _test()
