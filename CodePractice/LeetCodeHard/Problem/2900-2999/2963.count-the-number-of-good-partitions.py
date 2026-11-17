"""
LC 2963. Count the Number of Good Partitions — Hard
https://leetcode.com/problems/count-the-number-of-good-partitions/
Best: Merge by last occurrence → 2^(segments-1) — O(n)
"""

from typing import List

MOD = 10**9 + 7


class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        last = {}
        for i, x in enumerate(nums):
            last[x] = i
        ans = 1
        end = 0
        for i, x in enumerate(nums):
            end = max(end, last[x])
            if i == end and i != len(nums) - 1:
                ans = ans * 2 % MOD
        return ans


def _test():
    s = Solution()
    assert s.numberOfGoodPartitions([1, 2, 3, 4]) == 8
    assert s.numberOfGoodPartitions([1, 1, 1, 1]) == 1
    assert s.numberOfGoodPartitions([1, 2, 1, 3]) == 2
    print("LC 2963 OK")


if __name__ == "__main__":
    _test()
