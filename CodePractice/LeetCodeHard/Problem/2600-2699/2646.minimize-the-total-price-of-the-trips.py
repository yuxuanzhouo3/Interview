"""
LC 2646. Minimize the Total Price of the Trips — Hard
https://leetcode.com/problems/minimize-the-total-price-of-the-trips/
Best: Tree DP half price on path counts — O(n+trips)
"""

from typing import List
from collections import defaultdict


class Solution:
    def minimumTotalPrice(self, n: int, edges: List[List[int]], price: List[int], trips: List[List[int]]) -> int:
        g = [[] for _ in range(n)]
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)
        cnt = [0] * n

        def mark(start, end):
            parent = [-1] * n
            stack = [start]
            parent[start] = start
            while stack:
                u = stack.pop()
                if u == end:
                    break
                for v in g[u]:
                    if parent[v] == -1:
                        parent[v] = u
                        stack.append(v)
            u = end
            while True:
                cnt[u] += 1
                if u == start:
                    break
                u = parent[u]

        for a, b in trips:
            mark(a, b)

        # tree DP: dp[u][0]=not half, dp[u][1]=half u
        def dfs(u, p):
            full = price[u] * cnt[u]
            half = full // 2
            not_take = full
            take = half
            for v in g[u]:
                if v == p:
                    continue
                nt, tk = dfs(v, u)
                not_take += min(nt, tk)
                take += nt  # child cannot half if u halved
            return not_take, take

        return min(dfs(0, -1))


def _test():
    s = Solution()
    assert s.minimumTotalPrice(4, [[0,1],[1,2],[1,3]], [2,2,10,6], [[0,3],[2,1],[2,3]]) == 23
    assert s.minimumTotalPrice(2, [[0,1]], [2,2], [[0,0]]) == 1
    print("LC 2646 OK")


if __name__ == "__main__":
    _test()

