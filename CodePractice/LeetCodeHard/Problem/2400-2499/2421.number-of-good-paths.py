"""
LC 2421. Number of Good Paths — Hard
https://leetcode.com/problems/number-of-good-paths/
Best: Union-Find by increasing value — O(n α(n) + n log n)
"""

from typing import List
from collections import defaultdict


class Solution:
    def numberOfGoodPaths(self, vals: List[int], edges: List[List[int]]) -> int:
        n = len(vals)
        g = defaultdict(list)
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)

        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        nodes = defaultdict(list)
        for i, v in enumerate(vals):
            nodes[v].append(i)

        ans = n
        for v in sorted(nodes):
            for u in nodes[v]:
                for nei in g[u]:
                    if vals[nei] <= v:
                        pu, pv_ = find(u), find(nei)
                        if pu != pv_:
                            parent[pu] = pv_
            # count pairs within same component among nodes with value v
            cnt = defaultdict(int)
            for u in nodes[v]:
                cnt[find(u)] += 1
            for c in cnt.values():
                ans += c * (c - 1) // 2
        return ans


def _test():
    s = Solution()
    assert s.numberOfGoodPaths([1, 3, 2, 1, 3], [[0, 1], [0, 2], [2, 3], [2, 4]]) == 6
    assert s.numberOfGoodPaths([1, 1, 2, 2, 3], [[0, 1], [1, 2], [2, 3], [2, 4]]) == 7
    assert s.numberOfGoodPaths([1], []) == 1
    print("LC 2421 OK")


if __name__ == "__main__":
    _test()
