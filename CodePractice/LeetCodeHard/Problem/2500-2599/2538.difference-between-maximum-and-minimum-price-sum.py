"""
LC 2538. Difference Between Maximum and Minimum Price Sum — Hard
https://leetcode.com/problems/difference-between-maximum-and-minimum-price-sum/
Best: Tree DP top-2 down paths — O(n)
"""

from typing import List


class Solution:
    def maxOutput(self, n: int, edges: List[List[int]], price: List[int]) -> int:
        g = [[] for _ in range(n)]
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)
        ans = 0

        def dfs(u: int, p: int) -> int:
            nonlocal ans
            max1 = max2 = 0
            for v in g[u]:
                if v == p:
                    continue
                t = dfs(v, u)
                if t > max1:
                    max2, max1 = max1, t
                elif t > max2:
                    max2 = t
            ans = max(ans, max1 + max2)
            return max(max1, max2) + price[u]

        dfs(0, -1)
        return ans


def _test():
    s = Solution()
    assert s.maxOutput(6, [[0, 1], [1, 2], [1, 3], [3, 4], [3, 5]], [9, 8, 7, 6, 10, 5]) == 24
    assert s.maxOutput(3, [[0, 1], [1, 2]], [1, 1, 1]) == 2
    print("LC 2538 OK")


if __name__ == "__main__":
    _test()
