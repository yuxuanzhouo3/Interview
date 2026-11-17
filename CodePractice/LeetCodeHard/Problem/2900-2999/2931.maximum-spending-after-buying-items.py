"""
LC 2931. Maximum Spending After Buying Items — Hard
https://leetcode.com/problems/maximum-spending-after-buying-items/
Best: Always buy current cheapest available (min-heap of row ends) — O(mn log m)
"""

from typing import List
import heapq


class Solution:
    def maxSpending(self, values: List[List[int]]) -> int:
        m, n = len(values), len(values[0])
        # each row non-increasing; must buy from right (cheapest remaining)
        heap = []
        idx = [n - 1] * m
        for i in range(m):
            heapq.heappush(heap, (values[i][n - 1], i))
        day = 1
        ans = 0
        for _ in range(m * n):
            v, i = heapq.heappop(heap)
            ans += v * day
            day += 1
            idx[i] -= 1
            if idx[i] >= 0:
                heapq.heappush(heap, (values[i][idx[i]], i))
        return ans


def _test():
    s = Solution()
    assert s.maxSpending([[8, 5, 2], [6, 4, 1], [9, 7, 3]]) == 285
    assert s.maxSpending([[10, 8, 6, 4, 2], [9, 7, 5, 3, 2]]) == 386
    print("LC 2931 OK")


if __name__ == "__main__":
    _test()
