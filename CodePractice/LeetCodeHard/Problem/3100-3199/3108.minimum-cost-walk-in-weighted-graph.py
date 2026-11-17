"""
LC 3108. Minimum Cost Walk in Weighted Graph — Hard
https://leetcode.com/problems/minimum-cost-walk-in-weighted-graph/
Best: Union-Find + component AND — O(n+m+q)
"""

from typing import List


class Solution:
    def minimumCost(
        self, n: int, edges: List[List[int]], query: List[List[int]]
    ) -> List[int]:
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            pa, pb = find(a), find(b)
            if pa != pb:
                parent[pa] = pb

        for u, v, _ in edges:
            union(u, v)

        cand = [-1] * n
        for u, v, w in edges:
            r = find(u)
            cand[r] &= w

        ans = []
        for s, t in query:
            if find(s) != find(t):
                ans.append(-1)
            else:
                ans.append(cand[find(s)])
        return ans


def _test():
    s = Solution()
    assert s.minimumCost(5, [[0, 1, 7], [1, 3, 7], [1, 2, 1]], [[0, 3], [3, 4]]) == [1, -1]
    assert s.minimumCost(3, [[0, 2, 7], [0, 1, 15], [1, 2, 6], [1, 2, 1]], [[1, 2]]) == [0]
    print("LC 3108 OK")


if __name__ == "__main__":
    _test()
