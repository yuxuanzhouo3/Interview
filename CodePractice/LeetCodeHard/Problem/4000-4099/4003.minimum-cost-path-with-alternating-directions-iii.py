"""
LC 4003. Minimum Cost Path with Alternating Directions III — Hard
https://leetcode.com/problems/minimum-cost-path-with-alternating-directions-iii/
Best: doocs/leetcode reference solution (verified on examples)
"""

from math import inf
from heapq import heappush, heappop, heapify


class Solution:
    def minCost(self, m: int, n: int, penalty: list[list[int]]) -> int:
        dist = [[[inf] * 2 for _ in range(n)] for _ in range(m)]
        dist[0][0][1] = 1
        pq = [(1, 0, 0, 1)]
        dirs = ((-1, 0), (0, 1), (0, -1), (1, 0))
        while pq:
            d, i, j, k = heappop(pq)
            if i == m - 1 and j == n - 1:
                return d
            if d > dist[i][j][k]:
                continue

            p = penalty[i][j]
            nd = d + p
            if nd < dist[i][j][k ^ 1]:
                dist[i][j][k ^ 1] = nd
                heappush(pq, (nd, i, j, k ^ 1))

            for idx, (dx, dy) in enumerate(dirs):
                x, y = i + dx, j + dy
                if 0 <= x < m and 0 <= y < n:
                    nd = d + (x + 1) * (y + 1) + (idx & 1 ^ k) * p
                    if nd < dist[x][y][k ^ 1]:
                        dist[x][y][k ^ 1] = nd
                        heappush(pq, (nd, x, y, k ^ 1))

def _test():
    s = Solution()
    assert s.minCost(2, 2, [[5,3],[1,4]]) == 8
    assert s.minCost(2, 2, [[0,7],[3,2]]) == 7
    assert s.minCost(2, 3, [[8,0,9],[7,4,1]]) == 12
    print("LC 4003 OK")


if __name__ == "__main__":
    _test()
