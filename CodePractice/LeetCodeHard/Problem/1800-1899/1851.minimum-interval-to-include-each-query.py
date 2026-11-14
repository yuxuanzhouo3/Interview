"""
LC 1851. Minimum Interval to Include Each Query — Hard
https://leetcode.com/problems/minimum-interval-to-include-each-query/
Best: sort + min-heap by length — O((n+q) log(n+q))
"""

from typing import List
import heapq


class Solution:
    def minInterval(self, intervals: List[List[int]], queries: List[int]) -> List[int]:
        intervals.sort()
        qi = sorted(range(len(queries)), key=lambda i: queries[i])
        ans = [-1] * len(queries)
        heap = []  # (length, right)
        j = 0
        for i in qi:
            q = queries[i]
            while j < len(intervals) and intervals[j][0] <= q:
                l, r = intervals[j]
                heapq.heappush(heap, (r - l + 1, r))
                j += 1
            while heap and heap[0][1] < q:
                heapq.heappop(heap)
            if heap:
                ans[i] = heap[0][0]
        return ans


def _test():
    s = Solution()
    assert s.minInterval([[1, 4], [2, 4], [3, 6], [4, 4]], [2, 3, 4, 5]) == [3, 3, 1, 4]
    assert s.minInterval([[2, 3], [2, 5], [1, 8], [20, 25]], [2, 19, 5, 22]) == [2, -1, 4, 6]
    print("LC 1851 OK")


if __name__ == "__main__":
    _test()
