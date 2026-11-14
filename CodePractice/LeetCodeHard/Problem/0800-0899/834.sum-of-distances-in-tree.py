"""
LC 834. Sum of Distances in Tree — Hard
Best: Rerooting DP — O(n)
"""

from typing import List
from collections import defaultdict


class Solution:
    def sumOfDistancesInTree(self, n: int, edges: List[List[int]]) -> List[int]:
        g = defaultdict(list)
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)
        size = [1] * n
        ans = [0] * n

        def dfs1(u, p):
            for v in g[u]:
                if v == p:
                    continue
                dfs1(v, u)
                size[u] += size[v]
                ans[0] += size[v] + (ans[v] if False else 0)
            # fix: accumulate dist properly
        # rewrite cleaner
        dist0 = [0]

        def dfs_count(u, p):
            for v in g[u]:
                if v != p:
                    dfs_count(v, u)
                    size[u] += size[v]
                    dist0[0] += size[v]  # edges to subtree? wait need path lengths
        # standard solution:
        size = [0] * n
        dp = [0] * n

        def dfs1(u, p):
            size[u] = 1
            for v in g[u]:
                if v == p:
                    continue
                dfs1(v, u)
                size[u] += size[v]
                dp[u] += dp[v] + size[v]

        def dfs2(u, p):
            for v in g[u]:
                if v == p:
                    continue
                dp[v] = dp[u] - size[v] + (n - size[v])
                dfs2(v, u)

        dfs1(0, -1)
        dfs2(0, -1)
        return dp


def _test():
    s = Solution()
    assert s.sumOfDistancesInTree(6, [[0, 1], [0, 2], [2, 3], [2, 4], [2, 5]]) == [8, 12, 6, 10, 10, 10]
    print("LC 834 OK")


if __name__ == "__main__":
    _test()
