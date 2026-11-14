"""
LC 2448. Minimum Cost to Make Array Equal — Hard
https://leetcode.com/problems/minimum-cost-to-make-array-equal/
Best: Weighted median — O(n log n)
"""

from typing import List


class Solution:
    def minCost(self, nums: List[int], cost: List[int]) -> int:
        arr = sorted(zip(nums, cost))
        total = sum(cost)
        acc = 0
        target = None
        for v, c in arr:
            acc += c
            if acc * 2 >= total:
                target = v
                break
        return sum(abs(v - target) * c for v, c in zip(nums, cost))


def _test():
    s = Solution()
    assert s.minCost([1, 3, 5, 2], [2, 3, 1, 14]) == 8
    assert s.minCost([2, 2, 2, 2, 2], [4, 2, 8, 1, 3]) == 0
    print("LC 2448 OK")


if __name__ == "__main__":
    _test()
