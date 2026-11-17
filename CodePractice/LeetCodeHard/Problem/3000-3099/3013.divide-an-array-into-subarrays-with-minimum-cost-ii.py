"""
LC 3013. Divide an Array Into Subarrays With Minimum Cost II — Hard
https://leetcode.com/problems/divide-an-array-into-subarrays-with-minimum-cost-ii/
Best: Sliding window + two heaps / nsmallest — O(n*(dist) log)
"""


from typing import List
import heapq


class Solution:
    def minimumCost(self, nums: List[int], k: int, dist: int) -> int:
        n = len(nums)
        best = 10**30
        for r in range(1, n):
            L = max(1, r - dist)
            seg = nums[L : r + 1]
            if len(seg) < k - 1:
                continue
            best = min(best, nums[0] + sum(heapq.nsmallest(k - 1, seg)))
        return best


def _test():
    s = Solution()
    assert s.minimumCost([1, 3, 2, 6, 4, 2], 3, 3) == 5
    assert s.minimumCost([10, 1, 2, 2, 2, 1], 4, 3) == 15
    assert s.minimumCost([0, 1, 2, 3], 2, 3) == 1
    print("LC 3013 OK")


if __name__ == "__main__":
    _test()

