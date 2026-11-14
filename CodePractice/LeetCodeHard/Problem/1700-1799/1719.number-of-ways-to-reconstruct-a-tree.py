"""
LC 1719. Number of Ways to Reconstruct a Tree — Hard
https://leetcode.com/problems/number-of-ways-to-reconstruct-a-tree/
Best: degree-order parent check — O(n^2)
"""

from typing import List
from collections import defaultdict


class Solution:
    def checkWays(self, pairs: List[List[int]]) -> int:
        g = defaultdict(set)
        for a, b in pairs:
            g[a].add(b)
            g[b].add(a)
        nodes = sorted(g.keys(), key=lambda x: len(g[x]), reverse=True)
        n = len(nodes)
        if len(g[nodes[0]]) != n - 1:
            return 0
        ans = 1
        # ancestor[x] = True once processed (from root down by degree)
        for i, x in enumerate(nodes):
            if i == 0:
                continue
            # parent = neighbor with smallest degree among those already "possible"
            # All neighbors of x with larger/equal degree could be ancestors; pick minimal deg
            p = None
            pd = n + 1
            for y in g[x]:
                d = len(g[y])
                if d < pd:
                    pd = d
                    p = y
            # check: g[x] ⊆ g[p] ∪ {p}
            for y in g[x]:
                if y != p and y not in g[p]:
                    return 0
            if len(g[x]) == len(g[p]):
                ans = 2
        return ans


def _test():
    s = Solution()
    assert s.checkWays([[1, 2], [2, 3]]) == 1
    assert s.checkWays([[1, 2], [2, 3], [1, 3]]) == 2
    assert s.checkWays([[1, 2], [2, 3], [2, 4], [1, 5]]) == 0
    print("LC 1719 OK")


if __name__ == "__main__":
    _test()
