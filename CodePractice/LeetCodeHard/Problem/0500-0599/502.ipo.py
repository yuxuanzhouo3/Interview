"""
LC 502. IPO — Hard
https://leetcode.com/problems/ipo/
Best: Sort by capital + max-heap profits — O(n log n)
"""

from typing import List
import heapq


class Solution:
    def findMaximizedCapital(
        self, k: int, w: int, profits: List[int], capital: List[int]
    ) -> int:
        projects = sorted(zip(capital, profits))
        i = 0
        n = len(projects)
        heap = []  # max-heap of profits (-p)
        for _ in range(k):
            while i < n and projects[i][0] <= w:
                heapq.heappush(heap, -projects[i][1])
                i += 1
            if not heap:
                break
            w += -heapq.heappop(heap)
        return w


def _test():
    s = Solution()
    assert s.findMaximizedCapital(2, 0, [1, 2, 3], [0, 1, 1]) == 4
    assert s.findMaximizedCapital(3, 0, [1, 2, 3], [0, 1, 2]) == 6
    assert s.findMaximizedCapital(1, 0, [1, 2, 3], [1, 1, 2]) == 0
    print("LC 502 OK")


if __name__ == "__main__":
    _test()
