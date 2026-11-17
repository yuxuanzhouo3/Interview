"""
LC 2846. Minimum Edge Weight Equilibrium Queries in a Tree — Hard
https://leetcode.com/problems/minimum-edge-weight-equilibrium-queries-in-a-tree/
Best: LCA + weight frequency binary lifting — O((n+q) log n)
"""

from typing import List


class Solution:
    def minOperationsQueries(self, n: int, edges: List[List[int]], queries: List[List[int]]) -> List[int]:
        from collections import defaultdict
        g = [[] for _ in range(n)]
        for u, v, w in edges:
            g[u].append((v, w))
            g[v].append((u, w))
        LOG = n.bit_length()
        parent = [[-1] * n for _ in range(LOG)]
        depth = [0] * n
        cnt = [[0] * 27 for _ in range(n)]  # weight 1..26

        def dfs(u, p):
            for v, w in g[u]:
                if v == p:
                    continue
                depth[v] = depth[u] + 1
                parent[0][v] = u
                cnt[v] = cnt[u][:]
                cnt[v][w] += 1
                dfs(v, u)

        dfs(0, -1)
        for j in range(1, LOG):
            for i in range(n):
                if parent[j - 1][i] != -1:
                    parent[j][i] = parent[j - 1][parent[j - 1][i]]

        def lca(a, b):
            if depth[a] < depth[b]:
                a, b = b, a
            diff = depth[a] - depth[b]
            for j in range(LOG):
                if diff & (1 << j):
                    a = parent[j][a]
            if a == b:
                return a
            for j in range(LOG - 1, -1, -1):
                if parent[j][a] != parent[j][b]:
                    a = parent[j][a]
                    b = parent[j][b]
            return parent[0][a]

        ans = []
        for u, v in queries:
            a = lca(u, v)
            freq = [cnt[u][w] + cnt[v][w] - 2 * cnt[a][w] for w in range(27)]
            total = sum(freq)
            mx = max(freq) if total else 0
            ans.append(total - mx)
        return ans


def _test():
    s = Solution()
    assert s.minOperationsQueries(
        7,
        [[0, 1, 1], [1, 2, 1], [2, 3, 1], [3, 4, 2], [4, 5, 2], [5, 6, 2]],
        [[0, 3], [3, 6], [2, 6], [0, 6]],
    ) == [0, 0, 1, 3]
    assert s.minOperationsQueries(
        8,
        [[1, 2, 6], [1, 3, 4], [2, 4, 6], [2, 5, 3], [3, 6, 6], [3, 0, 8], [7, 0, 2]],
        [[4, 6], [0, 4], [6, 5], [7, 4]],
    ) == [1, 2, 2, 3]
    print("LC 2846 OK")


if __name__ == "__main__":
    _test()
