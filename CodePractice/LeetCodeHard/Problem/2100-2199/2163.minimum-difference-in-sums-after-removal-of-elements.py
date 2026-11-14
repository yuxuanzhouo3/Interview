"""
LC 2163. Minimum Difference in Sums After Removal of Elements — Hard
https://leetcode.com/problems/minimum-difference-in-sums-after-removal-of-elements/
Best: prefix min-heap + suffix max-heap — O(n log n)
"""

from typing import List
import heapq


class Solution:
    def minimumDifference(self, nums: List[int]) -> int:
        m = len(nums)
        n = m // 3
        # left_min[i]: min sum of n elems from nums[0..i] (i >= n-1), for i in [n-1, 2n-1]
        left = [0] * m
        pq = []
        s = 0
        for i in range(n):
            heapq.heappush(pq, -nums[i])
            s += nums[i]
        left[n - 1] = s
        for i in range(n, 2 * n):
            heapq.heappush(pq, -nums[i])
            s += nums[i]
            s -= -heapq.heappop(pq)
            left[i] = s
        # right_max[i]: max sum of n elems from nums[i..m-1]
        right = [0] * m
        pq = []
        s = 0
        for i in range(m - 1, m - n - 1, -1):
            heapq.heappush(pq, nums[i])
            s += nums[i]
        right[m - n] = s
        for i in range(m - n - 1, n - 1, -1):
            heapq.heappush(pq, nums[i])
            s += nums[i]
            s -= heapq.heappop(pq)
            right[i] = s
        ans = float("inf")
        for i in range(n - 1, 2 * n):
            ans = min(ans, left[i] - right[i + 1])
        return ans


def _test():
    s = Solution()
    assert s.minimumDifference([7, 9, 5, 8, 1, 3]) == 1
    assert s.minimumDifference([3, 1, 2]) == -1
    print("LC 2163 OK")


if __name__ == "__main__":
    _test()
