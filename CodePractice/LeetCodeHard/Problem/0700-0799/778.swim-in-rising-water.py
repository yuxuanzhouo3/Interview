"""
LC 778. Swim in Rising Water — Hard
Best: Dijkstra / binary search + BFS — O(n^2 log n)
"""

from typing import List
import heapq


class Solution:
    def swimInWater(self, grid: List[List[int]]) -> int:
        n = len(grid)
        heap = [(grid[0][0], 0, 0)]
        seen = {(0, 0)}
        while heap:
            t, i, j = heapq.heappop(heap)
            if i == j == n - 1:
                return t
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n and (ni, nj) not in seen:
                    seen.add((ni, nj))
                    heapq.heappush(heap, (max(t, grid[ni][nj]), ni, nj))
        return -1


def _test():
    s = Solution()
    assert s.swimInWater([[0, 2], [1, 3]]) == 3
    print("LC 778 OK")


if __name__ == "__main__":
    _test()
