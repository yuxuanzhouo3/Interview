"""
LC 3241. Time Taken to Mark All Nodes — Hard
https://leetcode.com/problems/time-taken-to-mark-all-nodes/
Best: Tree rerooting DP — O(n)
"""

from typing import List


class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)

        def cost(x: int) -> int:
            return 1 if x % 2 else 2

        best1 = [0] * n
        best2 = [0] * n

        def dfs1(u: int, p: int) -> None:
            for v in g[u]:
                if v == p:
                    continue
                dfs1(v, u)
                val = cost(v) + best1[v]
                if val > best1[u]:
                    best2[u], best1[u] = best1[u], val
                elif val > best2[u]:
                    best2[u] = val

        dfs1(0, -1)
        up = [0] * n

        def dfs2(u: int, p: int) -> None:
            for v in g[u]:
                if v == p:
                    continue
                other = best2[u] if cost(v) + best1[v] == best1[u] else best1[u]
                up[v] = cost(u) + max(up[u], other)
                dfs2(v, u)

        dfs2(0, -1)
        return [max(best1[i], up[i]) for i in range(n)]


def _test():
    s = Solution()
    assert s.timeTaken([[0, 1], [0, 2]]) == [2, 4, 3]
    assert s.timeTaken([[0, 1]]) == [1, 2]
    assert s.timeTaken([[2, 4], [0, 1], [2, 3], [2, 0]]) == [4, 6, 3, 5, 5]
    print("LC 3241 OK")


if __name__ == "__main__":
    _test()
