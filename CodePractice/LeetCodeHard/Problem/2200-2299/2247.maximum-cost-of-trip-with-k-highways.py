"""
LC 2247. Maximum Cost of Trip With K Highways — Hard
https://leetcode.com/problems/maximum-cost-of-trip-with-k-highways/
Best: Bitmask DP f[mask][end] — O(2^n * n^2), n<=15
"""
from typing import List
from collections import defaultdict
from math import inf


class Solution:
    def maximumCost(self, n: int, highways: List[List[int]], k: int) -> int:
        if k >= n:
            return -1
        g = defaultdict(list)
        for a, b, cost in highways:
            g[a].append((b, cost))
            g[b].append((a, cost))
        f = [[-inf] * n for _ in range(1 << n)]
        for i in range(n):
            f[1 << i][i] = 0
        ans = -1
        for mask in range(1 << n):
            for j in range(n):
                if mask >> j & 1:
                    for h, cost in g[j]:
                        if mask >> h & 1:
                            f[mask][j] = max(f[mask][j], f[mask ^ (1 << j)][h] + cost)
                if bin(mask).count("1") == k + 1:
                    ans = max(ans, f[mask][j])
        return ans


def _test():
    s = Solution()
    assert s.maximumCost(5, [[0,1,4],[2,1,3],[1,4,11],[3,2,3],[3,4,2]], 3) == 17
    assert s.maximumCost(4, [[0,1,3],[2,3,2]], 2) == -1
    print("LC 2247 OK")


if __name__ == "__main__":
    _test()
