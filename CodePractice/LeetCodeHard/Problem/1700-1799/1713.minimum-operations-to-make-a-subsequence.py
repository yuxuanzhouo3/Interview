"""
LC 1713. Minimum Operations to Make a Subsequence — Hard
https://leetcode.com/problems/minimum-operations-to-make-a-subsequence/
Best: map to LIS — O(n log n)
"""

from typing import List
import bisect


class Solution:
    def minOperations(self, target: List[int], arr: List[int]) -> int:
        pos = {v: i for i, v in enumerate(target)}
        seq = [pos[x] for x in arr if x in pos]
        tails = []
        for x in seq:
            j = bisect.bisect_left(tails, x)
            if j == len(tails):
                tails.append(x)
            else:
                tails[j] = x
        return len(target) - len(tails)


def _test():
    s = Solution()
    assert s.minOperations([5, 1, 3], [9, 4, 2, 3, 4]) == 2
    assert s.minOperations([6, 4, 8, 1, 3, 2], [4, 7, 6, 2, 3, 8, 6, 1]) == 3
    print("LC 1713 OK")


if __name__ == "__main__":
    _test()
