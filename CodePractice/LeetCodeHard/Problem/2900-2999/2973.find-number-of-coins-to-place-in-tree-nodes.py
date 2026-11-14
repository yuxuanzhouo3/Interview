"""
LC 2973. Find Number of Coins to Place in Tree Nodes — Hard
https://leetcode.com/problems/find-number-of-coins-to-place-in-tree-nodes/
Best: DFS keep top-3 max & top-2 min costs — O(n)
"""

from typing import List
from collections import defaultdict


class Solution:
    def placedCoins(self, edges: List[List[int]], cost: List[int]) -> List[int]:
        n = len(cost)
        g = defaultdict(list)
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)
        ans = [0] * n

        def dfs(u: int, p: int):
            vals = [cost[u]]
            for v in g[u]:
                if v == p:
                    continue
                vals.extend(dfs(v, u))
            vals.sort()
            sz = len(vals)
            if sz < 3:
                ans[u] = 1
            else:
                cand = vals[-1] * vals[-2] * vals[-3]
                cand2 = vals[0] * vals[1] * vals[-1]
                ans[u] = max(0, cand, cand2)
            if sz > 5:
                vals = vals[:2] + vals[-3:]
            return vals

        dfs(0, -1)
        return ans


def _test():
    s = Solution()
    assert s.placedCoins([[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]], [1, 2, 3, 4, 5, 6]) == [120, 1, 1, 1, 1, 1]
    assert s.placedCoins([[0, 1], [0, 2], [1, 3], [1, 4], [1, 5], [2, 6], [2, 7], [2, 8]], [1, 4, 2, 3, 5, 7, 8, -4, 2]) == [280, 140, 32, 1, 1, 1, 1, 1, 1]
    assert s.placedCoins([[0, 1], [0, 2]], [1, 2, -2]) == [0, 1, 1]
    print("LC 2973 OK")


if __name__ == "__main__":
    _test()
