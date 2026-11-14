"""
LC 1192. Critical Connections in a Network — Hard
https://leetcode.com/problems/critical-connections-in-a-network/
Best: Tarjan bridges — O(n + e)
"""

from typing import List


class Solution:
    def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        g = [[] for _ in range(n)]
        for a, b in connections:
            g[a].append(b)
            g[b].append(a)
        disc = [-1] * n
        low = [0] * n
        time = 0
        bridges = []

        def dfs(u, parent):
            nonlocal time
            disc[u] = low[u] = time
            time += 1
            for v in g[u]:
                if v == parent:
                    continue
                if disc[v] == -1:
                    dfs(v, u)
                    low[u] = min(low[u], low[v])
                    if low[v] > disc[u]:
                        bridges.append([u, v])
                else:
                    low[u] = min(low[u], disc[v])

        dfs(0, -1)
        return bridges


def _test():
    s = Solution()
    r = s.criticalConnections(4, [[0, 1], [1, 2], [2, 0], [1, 3]])
    assert sorted(sorted(e) for e in r) == [[1, 3]]
    r2 = s.criticalConnections(2, [[0, 1]])
    assert sorted(sorted(e) for e in r2) == [[0, 1]]
    print("LC 1192 OK")


if __name__ == "__main__":
    _test()
