"""
LC 1121. Divide Array Into Increasing Sequences — Hard
https://leetcode.com/problems/divide-array-into-increasing-sequences/
Tags: Array, Counting
Best: 最大频次 ≤ ⌊n/K⌋ 的序列数约束
"""

from typing import List
from collections import Counter


class Solution:
    def canDivideIntoSubsequences(self, nums: List[int], K: int) -> bool:
        # nums is non-decreasing; same value cannot share a strictly increasing seq
        # need at least max_freq sequences, each length >= K => max_freq * K <= n
        n = len(nums)
        max_freq = max(Counter(nums).values())
        return max_freq * K <= n


def _test():
    sol = Solution()
    assert sol.canDivideIntoSubsequences([1, 2, 2, 3, 3, 4, 4], 3) is True
    assert sol.canDivideIntoSubsequences([5, 6, 6, 7, 8], 3) is False
    print("LC 1121 OK")


if __name__ == "__main__":
    _test()

