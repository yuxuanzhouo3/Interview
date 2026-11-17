"""
LC 2608. Shortest Cycle in a Graph — Hard
https://leetcode.com/problems/shortest-cycle-in-a-graph/
Best: BFS from every node — O(n(n+m))
"""

from typing import List
from collections import deque


class Solution:
    def findShortestCycle(self, n: int, edges: List[List[int]]) -> int:
        g = [[] for _ in range(n)]
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)
        ans = float("inf")
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in g[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif parent[u] != v:
                        ans = min(ans, dist[u] + dist[v] + 1)
        return -1 if ans == float("inf") else ans


def _test():
    s = Solution()
    assert s.findShortestCycle(7, [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 6], [6, 3]]) == 3
    assert s.findShortestCycle(4, [[0, 1], [0, 2]]) == -1
    print("LC 2608 OK")


if __name__ == "__main__":
    _test()
