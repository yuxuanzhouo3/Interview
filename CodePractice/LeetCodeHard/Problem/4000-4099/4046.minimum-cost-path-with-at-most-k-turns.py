"""
LC 4046. Minimum Cost Path with at Most K Turns — Hard
https://leetcode.com/problems/minimum-cost-path-with-at-most-k-turns/
Best: Dijkstra on (r,c,dir,turns)
"""
import heapq


class Solution:
    def minCost(self, grid: list[list[int]], k: int) -> int:
        m, n = len(grid), len(grid[0])
        if m == 1 and n == 1:
            return grid[0][0]
        dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
        INF = 10**18
        # dist[r][c][dir][turns] — dir in 0..3, turns 0..k
        dist = [[[[INF] * (k + 1) for _ in range(4)] for _ in range(n)] for _ in range(m)]
        pq = []
        for d, (dr, dc) in enumerate(dirs):
            nr, nc = dr, dc
            if 0 <= nr < m and 0 <= nc < n:
                cost = grid[0][0] + grid[nr][nc]
                dist[nr][nc][d][0] = cost
                heapq.heappush(pq, (cost, nr, nc, d, 0))
        while pq:
            cost, r, c, d, t = heapq.heappop(pq)
            if cost > dist[r][c][d][t]:
                continue
            if r == m - 1 and c == n - 1:
                return cost
            for nd, (dr, dc) in enumerate(dirs):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < m and 0 <= nc < n):
                    continue
                nt = t + (nd != d)
                if nt > k:
                    continue
                ncost = cost + grid[nr][nc]
                if ncost < dist[nr][nc][nd][nt]:
                    dist[nr][nc][nd][nt] = ncost
                    heapq.heappush(pq, (ncost, nr, nc, nd, nt))
        return -1


def _test():
    s = Solution()
    assert s.minCost([[2, 7, 3], [1, 4, 5]], 1) == 12
    assert s.minCost([[4, 1, 9], [3, 2, 5], [4, 8, 6]], 2) == 20
    assert s.minCost([[1, 9], [3, 4]], 0) == -1
    print("LC 4046 OK")


if __name__ == "__main__":
    _test()
