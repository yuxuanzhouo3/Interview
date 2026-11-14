"""
LC 2872. Maximum Number of K-Divisible Components — Hard
https://leetcode.com/problems/maximum-number-of-k-divisible-components/
Best: Tree DFS subtree sum % k — O(n)
"""

from typing import List
from collections import defaultdict


class Solution:
    def maxKDivisibleComponents(self, n: int, edges: List[List[int]], values: List[int], k: int) -> int:
        g = defaultdict(list)
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)
        ans = 0

        def dfs(u: int, p: int) -> int:
            nonlocal ans
            s = values[u]
            for v in g[u]:
                if v != p:
                    s += dfs(v, u)
            if s % k == 0:
                ans += 1
                return 0
            return s

        dfs(0, -1)
        return ans


def _test():
    s = Solution()
    assert s.maxKDivisibleComponents(5, [[0, 2], [1, 2], [1, 3], [2, 4]], [1, 8, 1, 4, 4], 6) == 2
    assert s.maxKDivisibleComponents(7, [[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6]], [3, 0, 6, 1, 5, 2, 1], 3) == 3
    print("LC 2872 OK")


if __name__ == "__main__":
    _test()
