"""
LC 2581. Count Number of Possible Root Nodes — Hard
https://leetcode.com/problems/count-number-of-possible-root-nodes/
Best: Reroot DP with guess set — O(n+|guesses|)
"""

from typing import List
from collections import defaultdict


class Solution:
    def rootCount(self, edges: List[List[int]], guesses: List[List[int]], k: int) -> int:
        n = len(edges) + 1
        g = [[] for _ in range(n)]
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)
        gs = {(a, b) for a, b in guesses}

        # count correct guesses if root=0
        def dfs(u, p):
            c = 0
            for v in g[u]:
                if v == p:
                    continue
                if (u, v) in gs:
                    c += 1
                c += dfs(v, u)
            return c

        base = dfs(0, -1)
        ans = 0

        def reroot(u, p, cur):
            nonlocal ans
            if cur >= k:
                ans += 1
            for v in g[u]:
                if v == p:
                    continue
                nxt = cur
                if (u, v) in gs:
                    nxt -= 1
                if (v, u) in gs:
                    nxt += 1
                reroot(v, u, nxt)

        reroot(0, -1, base)
        return ans


def _test():
    s = Solution()
    assert s.rootCount([[0,1],[1,2],[1,3],[4,2]], [[1,3],[0,1],[1,0],[2,4]], 3) == 3
    assert s.rootCount([[0,1],[1,2],[2,3],[3,4]], [[1,0],[3,4],[2,1],[3,2]], 1) == 5
    print("LC 2581 OK")


if __name__ == "__main__":
    _test()

