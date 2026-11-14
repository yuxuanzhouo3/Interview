"""
LC 1857. Largest Color Value in a Directed Graph — Hard
https://leetcode.com/problems/largest-color-value-in-a-directed-graph/
Best: topo + DP counts — O(V+E)
"""

from typing import List
from collections import deque, defaultdict


class Solution:
    def largestPathValue(self, colors: str, edges: List[List[int]]) -> int:
        n = len(colors)
        g = [[] for _ in range(n)]
        indeg = [0] * n
        for u, v in edges:
            g[u].append(v)
            indeg[v] += 1
        dp = [[0] * 26 for _ in range(n)]
        q = deque()
        for i in range(n):
            if indeg[i] == 0:
                q.append(i)
        seen = 0
        ans = 0
        while q:
            u = q.popleft()
            seen += 1
            c = ord(colors[u]) - 97
            dp[u][c] += 1
            ans = max(ans, dp[u][c])
            for v in g[u]:
                for k in range(26):
                    if dp[u][k] > dp[v][k]:
                        dp[v][k] = dp[u][k]
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        return ans if seen == n else -1


def _test():
    s = Solution()
    assert s.largestPathValue("abaca", [[0, 1], [0, 2], [2, 3], [3, 4]]) == 3
    assert s.largestPathValue("a", [[0, 0]]) == -1
    print("LC 1857 OK")


if __name__ == "__main__":
    _test()
