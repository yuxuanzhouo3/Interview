"""
LC 1627. Graph Connectivity With Threshold — Hard
https://leetcode.com/problems/graph-connectivity-with-threshold/
Best: UnionFind on multiples of d>threshold — O(n log n + q α(n))
"""

from typing import List


class Solution:
    def areConnected(self, n: int, threshold: int, queries: List[List[int]]) -> List[bool]:
        parent = list(range(n + 1))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for d in range(threshold + 1, n + 1):
            for multiple in range(2 * d, n + 1, d):
                union(d, multiple)

        return [find(a) == find(b) for a, b in queries]


def _test():
    s = Solution()
    assert s.areConnected(6, 2, [[1, 4], [2, 5], [3, 6]]) == [False, False, True]
    assert s.areConnected(6, 0, [[4, 5], [3, 4], [3, 2], [2, 6], [1, 3]]) == [
        True,
        True,
        True,
        True,
        True,
    ]
    assert s.areConnected(5, 1, [[4, 5], [4, 5], [3, 2], [2, 3], [3, 4]]) == [
        False,
        False,
        False,
        False,
        False,
    ]
    print("LC 1627 OK")


if __name__ == "__main__":
    _test()
