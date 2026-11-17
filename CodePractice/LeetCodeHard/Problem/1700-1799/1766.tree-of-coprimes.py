"""
LC 1766. Tree of Coprimes — Hard
https://leetcode.com/problems/tree-of-coprimes/
Best: DFS with last-seen stack per value — O(n · V)
"""

from math import gcd
from typing import List


class Solution:
    def getCoprimes(self, nums: List[int], edges: List[List[int]]) -> List[int]:
        n = len(nums)
        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        ans = [-1] * n
        # last[val] = list of (node, depth)
        last = [[] for _ in range(51)]

        def dfs(u, parent, depth):
            best_d = -1
            best_node = -1
            for v in range(1, 51):
                if last[v] and gcd(nums[u], v) == 1:
                    node, d = last[v][-1]
                    if d > best_d:
                        best_d = d
                        best_node = node
            ans[u] = best_node
            last[nums[u]].append((u, depth))
            for v in g[u]:
                if v != parent:
                    dfs(v, u, depth + 1)
            last[nums[u]].pop()

        dfs(0, -1, 0)
        return ans


def _test():
    s = Solution()
    assert s.getCoprimes([2, 3, 3, 2], [[0, 1], [1, 2], [1, 3]]) == [-1, 0, 0, 1]
    assert s.getCoprimes([5, 6, 10, 2, 3, 6, 15], [[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6]]) == [
        -1, 0, -1, 0, 0, 0, -1
    ]
    print("LC 1766 OK")


if __name__ == "__main__":
    _test()
