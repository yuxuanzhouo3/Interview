"""
LC 1787. Make the XOR of All Segments Equal to Zero — Hard
https://leetcode.com/problems/make-the-xor-of-all-segments-equal-to-zero/
Best: Group by i%k + DP on XOR — O(k · 2^10 · |vals|)
"""

from collections import Counter
from typing import List


class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # groups[r] = Counter of nums[i] for i%k==r
        groups = [Counter() for _ in range(k)]
        sizes = [0] * k
        for i, x in enumerate(nums):
            groups[i % k][x] += 1
            sizes[i % k] += 1

        INF = 10**9
        MA = 1 << 10
        # dp[xor] = min changes for first r groups to have XOR = xor
        dp = [INF] * MA
        dp[0] = 0
        for r in range(k):
            ndp = [INF] * MA
            total = sizes[r]
            # min changes if change all to some value (even not in group)
            min_prev = min(dp)
            # option1: set to a value that appears
            for xor in range(MA):
                if dp[xor] == INF:
                    continue
                for val, cnt in groups[r].items():
                    nxor = xor ^ val
                    ndp[nxor] = min(ndp[nxor], dp[xor] + total - cnt)
            # option2: set all positions to a brand-new value (any)
            # then changes = total, and can reach any new xor from prev
            for xor in range(MA):
                ndp[xor] = min(ndp[xor], min_prev + total)
            dp = ndp
        return dp[0]


def _test():
    s = Solution()
    assert s.minChanges([1, 2, 0, 3, 0], 1) == 3
    assert s.minChanges([3, 4, 5, 2, 1, 7, 3, 4, 7], 3) == 3
    assert s.minChanges([1, 2, 4, 1, 2, 5, 1, 2, 6], 3) == 3
    print("LC 1787 OK")


if __name__ == "__main__":
    _test()
