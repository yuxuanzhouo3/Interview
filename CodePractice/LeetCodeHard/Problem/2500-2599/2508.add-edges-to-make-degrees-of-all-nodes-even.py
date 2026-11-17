"""
LC 2508. Add Edges to Make Degrees of All Nodes Even — Hard
https://leetcode.com/problems/add-edges-to-make-degrees-of-all-nodes-even/
Best: Classify 0/2/4 odd-degree nodes — O(n+m)
"""

from typing import List
from collections import defaultdict


class Solution:
    def isPossible(self, n: int, edges: List[List[int]]) -> bool:
        g = defaultdict(set)
        for a, b in edges:
            g[a].add(b)
            g[b].add(a)
        odd = [i for i in range(1, n + 1) if len(g[i]) % 2]
        if len(odd) == 0:
            return True
        if len(odd) == 2:
            a, b = odd
            if b not in g[a]:
                return True
            for c in range(1, n + 1):
                if c != a and c != b and c not in g[a] and c not in g[b]:
                    return True
            return False
        if len(odd) == 4:
            a, b, c, d = odd
            return (
                (b not in g[a] and d not in g[c])
                or (c not in g[a] and d not in g[b])
                or (d not in g[a] and c not in g[b])
            )
        return False


def _test():
    s = Solution()
    assert s.isPossible(5, [[1, 2], [2, 3], [3, 4], [4, 2], [1, 4], [2, 5]]) is True
    assert s.isPossible(4, [[1, 2], [3, 4]]) is True
    assert s.isPossible(4, [[1, 2], [1, 3], [1, 4]]) is False
    print("LC 2508 OK")


if __name__ == "__main__":
    _test()
