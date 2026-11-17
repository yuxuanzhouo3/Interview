"""
LC 1655. Distribute Repeating Integers — Hard
https://leetcode.com/problems/distribute-repeating-integers/
Best: Frequency + subset DP — O(n · 3^m)
"""

from collections import Counter
from typing import List


class Solution:
    def canDistribute(self, nums: List[int], quantity: List[int]) -> bool:
        freq = sorted(Counter(nums).values(), reverse=True)
        m = len(quantity)
        # only need top m frequencies
        freq = freq[:m]
        n = len(freq)
        # precompute subset sums of quantity
        N = 1 << m
        ssum = [0] * N
        for mask in range(1, N):
            b = mask.bit_length() - 1
            ssum[mask] = ssum[mask ^ (1 << b)] + quantity[b]

        dp = [False] * N
        dp[0] = True
        for f in freq:
            ndp = dp[:]
            for mask in range(N):
                if not dp[mask]:
                    continue
                left = (N - 1) ^ mask
                sub = left
                while sub:
                    if ssum[sub] <= f:
                        ndp[mask | sub] = True
                    sub = (sub - 1) & left
            dp = ndp
            if dp[N - 1]:
                return True
        return dp[N - 1]


def _test():
    s = Solution()
    assert s.canDistribute([1, 2, 3, 4], [2]) is False
    assert s.canDistribute([1, 2, 3, 3], [2]) is True
    assert s.canDistribute([1, 1, 2, 2], [2, 2]) is True
    print("LC 1655 OK")


if __name__ == "__main__":
    _test()
