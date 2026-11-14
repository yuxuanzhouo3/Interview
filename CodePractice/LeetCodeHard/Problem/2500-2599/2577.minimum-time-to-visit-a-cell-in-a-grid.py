"""
LC 2577. Minimum Time to Visit a Cell In a Grid — Hard
https://leetcode.com/problems/minimum-time-to-visit-a-cell-in-a-grid/
Best: Dijkstra with parity wait — O(mn log mn)
"""

from typing import List
import heapq


class Solution:
    def minimumTime(self, grid: List[List[int]]) -> int:
        if grid[0][1] > 1 and grid[1][0] > 1:
            return -1
        m, n = len(grid), len(grid[0])
        dist = [[10**18] * n for _ in range(m)]
        dist[0][0] = 0
        pq = [(0, 0, 0)]
        while pq:
            t, i, j = heapq.heappop(pq)
            if t > dist[i][j]:
                continue
            if i == m - 1 and j == n - 1:
                return t
            for ni, nj in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= ni < m and 0 <= nj < n:
                    wait = 0
                    if t + 1 < grid[ni][nj]:
                        wait = grid[ni][nj] - (t + 1)
                        if wait % 2:
                            wait += 1
                    nt = t + 1 + wait
                    if nt < dist[ni][nj]:
                        dist[ni][nj] = nt
                        heapq.heappush(pq, (nt, ni, nj))
        return -1


def _test():
    s = Solution()
    assert s.minimumTime([[0, 1, 3, 2], [5, 1, 2, 5], [4, 3, 8, 6]]) == 7
    assert s.minimumTime([[0, 2, 4], [3, 2, 1], [1, 0, 4]]) == -1
    print("LC 2577 OK")


if __name__ == "__main__":
    _test()
