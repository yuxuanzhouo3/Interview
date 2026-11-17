"""
LC 2493. Divide Nodes Into the Maximum Number of Groups — Hard
https://leetcode.com/problems/divide-nodes-into-the-maximum-number-of-groups/
Best: Bipartite check + BFS max depth per component — O(n(n+m))
"""

from typing import List
from collections import deque, defaultdict


class Solution:
    def magnificentSets(self, n: int, edges: List[List[int]]) -> int:
        g = [[] for _ in range(n + 1)]
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)

        color = [-1] * (n + 1)

        def bipartite(start):
            color[start] = 0
            q = deque([start])
            nodes = [start]
            while q:
                u = q.popleft()
                for v in g[u]:
                    if color[v] == -1:
                        color[v] = color[u] ^ 1
                        q.append(v)
                        nodes.append(v)
                    elif color[v] == color[u]:
                        return None
            return nodes

        def bfs_depth(src, nodes_set):
            dist = {src: 1}
            q = deque([src])
            mx = 1
            while q:
                u = q.popleft()
                for v in g[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        mx = max(mx, dist[v])
                        q.append(v)
            return mx

        ans = 0
        for i in range(1, n + 1):
            if color[i] != -1:
                continue
            nodes = bipartite(i)
            if nodes is None:
                return -1
            nset = set(nodes)
            best = 0
            for u in nodes:
                best = max(best, bfs_depth(u, nset))
            ans += best
        return ans


def _test():
    s = Solution()
    assert s.magnificentSets(6, [[1, 2], [1, 4], [1, 5], [2, 6], [2, 3], [4, 6]]) == 4
    assert s.magnificentSets(3, [[1, 2], [2, 3], [3, 1]]) == -1
    print("LC 2493 OK")


if __name__ == "__main__":
    _test()
