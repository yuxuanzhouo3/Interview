"""
LC 407. Trapping Rain Water II — Hard
Best: Min-heap BFS from border — O(mn log mn)
"""

from typing import List
import heapq


class Solution:
    def trapRainWater(self, heightMap: List[List[int]]) -> int:
        if not heightMap or not heightMap[0]:
            return 0
        m, n = len(heightMap), len(heightMap[0])
        vis = [[False] * n for _ in range(m)]
        heap = []
        for i in range(m):
            for j in range(n):
                if i in (0, m - 1) or j in (0, n - 1):
                    heapq.heappush(heap, (heightMap[i][j], i, j))
                    vis[i][j] = True
        ans = 0
        while heap:
            h, i, j = heapq.heappop(heap)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and not vis[ni][nj]:
                    vis[ni][nj] = True
                    ans += max(0, h - heightMap[ni][nj])
                    heapq.heappush(heap, (max(h, heightMap[ni][nj]), ni, nj))
        return ans


def _test():
    s = Solution()
    assert s.trapRainWater([[1, 4, 3, 1, 3, 2], [3, 2, 1, 3, 2, 4], [2, 3, 3, 2, 3, 1]]) == 4
    print("LC 407 OK")


if __name__ == "__main__":
    _test()
