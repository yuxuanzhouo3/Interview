"""
LC 685. Redundant Connection II — Hard
https://leetcode.com/problems/redundant-connection-ii/
Best: union-find cases (two parents / cycle) — O(n)
"""

from typing import List


class Solution:
    def findRedundantDirectedConnection(self, edges: List[List[int]]) -> List[int]:
        n = len(edges)
        parent = list(range(n + 1))
        cand1 = cand2 = None
        parents = [0] * (n + 1)

        for u, v in edges:
            if parents[v]:
                cand1 = [parents[v], v]
                cand2 = [u, v]
            else:
                parents[v] = u

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        parent = list(range(n + 1))
        for u, v in edges:
            if cand2 and [u, v] == cand2:
                continue  # skip second edge to double-parent node
            pu, pv = find(u), find(v)
            if pu == pv:
                # cycle
                if cand1:
                    return cand1
                return [u, v]
            parent[pv] = pu
        return cand2


def _test():
    s = Solution()
    assert s.findRedundantDirectedConnection([[1, 2], [1, 3], [2, 3]]) == [2, 3]
    assert s.findRedundantDirectedConnection([[1, 2], [2, 3], [3, 4], [4, 1], [1, 5]]) == [4, 1]
    assert s.findRedundantDirectedConnection([[2, 1], [3, 1], [4, 2], [1, 4]]) == [2, 1]
    print("LC 685 OK")


if __name__ == "__main__":
    _test()
