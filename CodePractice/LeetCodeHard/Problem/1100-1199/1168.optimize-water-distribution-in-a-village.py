"""
LC 1168. Optimize Water Distribution in a Village — Hard
https://leetcode.com/problems/optimize-water-distribution-in-a-village/
Tags: Union-Find, MST
Best: 虚拟水源点 0 + Kruskal MST
"""

from typing import List


class Solution:
    def minCostToSupplyWater(self, n: int, wells: List[int], pipes: List[List[int]]) -> int:
        edges = []
        for i, w in enumerate(wells, 1):
            edges.append((w, 0, i))
        for a, b, c in pipes:
            edges.append((c, a, b))
        edges.sort()
        parent = list(range(n + 1))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        ans = 0
        used = 0
        for c, u, v in edges:
            ru, rv = find(u), find(v)
            if ru == rv:
                continue
            parent[ru] = rv
            ans += c
            used += 1
            if used == n:
                break
        return ans


def _test():
    sol = Solution()
    assert sol.minCostToSupplyWater(3, [1, 2, 2], [[1, 2, 1], [2, 3, 1]]) == 3
    assert sol.minCostToSupplyWater(2, [1, 1], [[1, 2, 1], [1, 2, 2]]) == 2
    print("LC 1168 OK")


if __name__ == "__main__":
    _test()

