"""
LC 1575. Count All Possible Routes — Hard
https://leetcode.com/problems/count-all-possible-routes/
Best: Memo DFS on (city, fuel) — O(n^2 · fuel)
"""

from functools import lru_cache
from typing import List


class Solution:
    def countRoutes(self, locations: List[int], start: int, finish: int, fuel: int) -> int:
        MOD = 10**9 + 7
        n = len(locations)

        @lru_cache(None)
        def dfs(i: int, f: int) -> int:
            ans = 1 if i == finish else 0
            for j in range(n):
                if j == i:
                    continue
                cost = abs(locations[i] - locations[j])
                if cost <= f:
                    ans = (ans + dfs(j, f - cost)) % MOD
            return ans

        return dfs(start, fuel)


def _test():
    s = Solution()
    assert s.countRoutes([2, 3, 6, 8, 4], 1, 3, 5) == 4
    assert s.countRoutes([4, 3, 1], 1, 0, 6) == 5
    assert s.countRoutes([5, 2, 1], 0, 2, 3) == 0
    print("LC 1575 OK")


if __name__ == "__main__":
    _test()
