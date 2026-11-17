"""
LC 1548. The Most Similar Path in a Graph — Hard
Best: DP path matching target names — O(n * m * E)
"""

from typing import List


class Solution:
    def mostSimilar(self, n: int, roads: List[List[int]], names: List[str], targetPath: List[str]) -> List[int]:
        g = [[] for _ in range(n)]
        for a, b in roads:
            g[a].append(b)
            g[b].append(a)
        m = len(targetPath)
        INF = 10**9
        dp = [[INF] * n for _ in range(m)]
        pre = [[-1] * n for _ in range(m)]
        for u in range(n):
            dp[0][u] = 0 if names[u] == targetPath[0] else 1
        for i in range(1, m):
            for v in range(n):
                cost = 0 if names[v] == targetPath[i] else 1
                for u in g[v]:
                    if dp[i - 1][u] + cost < dp[i][v]:
                        dp[i][v] = dp[i - 1][u] + cost
                        pre[i][v] = u
        end = min(range(n), key=lambda x: dp[m - 1][x])
        path = [0] * m
        path[-1] = end
        for i in range(m - 1, 0, -1):
            path[i - 1] = pre[i][path[i]]
        return path


def _test():
    s = Solution()
    r = s.mostSimilar(3, [[0, 1], [1, 2], [0, 2]], ["A", "B", "C"], ["A", "B", "A"])
    assert len(r) == 3
    print("LC 1548 OK")


if __name__ == "__main__":
    _test()
