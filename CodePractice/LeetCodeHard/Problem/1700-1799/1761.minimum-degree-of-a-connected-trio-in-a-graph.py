"""
LC 1761. Minimum Degree of a Connected Trio in a Graph — Hard
https://leetcode.com/problems/minimum-degree-of-a-connected-trio-in-a-graph/
Best: Adjacency matrix enumerate triangles — O(n^3) / O(m^{1.5})
"""

from typing import List


class Solution:
    def minTrioDegree(self, n: int, edges: List[List[int]]) -> int:
        g = [[False] * (n + 1) for _ in range(n + 1)]
        deg = [0] * (n + 1)
        for u, v in edges:
            g[u][v] = g[v][u] = True
            deg[u] += 1
            deg[v] += 1
        ans = float("inf")
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                if not g[i][j]:
                    continue
                for k in range(j + 1, n + 1):
                    if g[i][k] and g[j][k]:
                        ans = min(ans, deg[i] + deg[j] + deg[k] - 6)
        return ans if ans < float("inf") else -1


def _test():
    s = Solution()
    assert s.minTrioDegree(6, [[1, 2], [1, 3], [3, 2], [4, 1], [5, 2], [3, 6]]) == 3
    assert s.minTrioDegree(7, [[1, 3], [4, 1], [4, 3], [2, 5], [5, 6], [6, 7], [7, 5], [2, 6]]) == 0
    print("LC 1761 OK")


if __name__ == "__main__":
    _test()
