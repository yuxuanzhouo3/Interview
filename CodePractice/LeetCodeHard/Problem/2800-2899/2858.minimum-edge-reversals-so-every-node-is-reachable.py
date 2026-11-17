"""
LC 2858. Minimum Edge Reversals So Every Node Is Reachable — Hard
https://leetcode.com/problems/minimum-edge-reversals-so-every-node-is-reachable/
Best: Reroot edge reversal count — O(n)
"""

from typing import List


class Solution:
    def minEdgeReversals(self, n: int, edges: List[List[int]]) -> List[int]:
        g = [[] for _ in range(n)]  # (to, cost) cost=0 if original u->v, 1 if need reverse
        for a, b in edges:
            g[a].append((b, 0))
            g[b].append((a, 1))
        ans = [0] * n

        def dfs(u, p):
            s = 0
            for v, c in g[u]:
                if v == p:
                    continue
                s += c + dfs(v, u)
            return s

        ans[0] = dfs(0, -1)

        def reroot(u, p):
            for v, c in g[u]:
                if v == p:
                    continue
                # move root u -> v: lose c (for edge u-v), gain 1-c
                ans[v] = ans[u] - c + (1 - c)
                reroot(v, u)

        reroot(0, -1)
        return ans


def _test():
    s = Solution()
    assert s.minEdgeReversals(4, [[2,0],[2,1],[1,3]]) == [1, 1, 0, 2]
    assert s.minEdgeReversals(3, [[1,2],[2,0]]) == [2, 0, 1]
    print("LC 2858 OK")


if __name__ == "__main__":
    _test()

