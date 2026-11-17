"""
LC 3797. Count Routes to Climb a Rectangular Grid — Hard
https://leetcode.com/problems/count-routes-to-climb-a-rectangular-grid/
Best: Row-by-row DP with at most one horizontal hop — O(n m^2)
"""

from __future__ import annotations

from typing import List


class Solution:
    def numberOfRoutes(self, grid: List[str], d: int) -> int:
        MOD = 10**9 + 7
        n, m = len(grid), len(grid[0])
        d2 = d * d
        rows = [[j for j in range(m) if grid[i][j] == "."] for i in range(n)]
        if not rows[n - 1]:
            return 0

        # dp[c] = ways to stand on column c of current row, ready to move up or finish
        dp = [0] * m
        for j in rows[n - 1]:
            dp[j] = 1

        ans = 0
        for r in range(n - 1, -1, -1):
            cols = rows[r]
            if r == 0:
                # can finish at any current cell
                for c in cols:
                    ans = (ans + dp[c]) % MOD
                # can do one horizontal then finish
                for i, c1 in enumerate(cols):
                    if dp[c1] == 0:
                        continue
                    for c2 in cols:
                        if c1 == c2:
                            continue
                        if (c1 - c2) * (c1 - c2) <= d2:
                            ans = (ans + dp[c1]) % MOD
                break

            # move to row r-1
            up = rows[r - 1]
            ndp = [0] * m
            if not up:
                return 0
            for c in cols:
                if dp[c] == 0:
                    continue
                # direct up
                for uc in up:
                    if (1 + (c - uc) * (c - uc)) <= d2:  # (r-(r-1))^2=1
                        ndp[uc] = (ndp[uc] + dp[c]) % MOD
                # horizontal then up
                for c2 in cols:
                    if c2 == c:
                        continue
                    if (c - c2) * (c - c2) > d2:
                        continue
                    for uc in up:
                        if (1 + (c2 - uc) * (c2 - uc)) <= d2:
                            ndp[uc] = (ndp[uc] + dp[c]) % MOD
            dp = ndp
        return ans % MOD


def _test():
    s = Solution()
    assert s.numberOfRoutes(["..", "#."], 1) == 2
    assert s.numberOfRoutes(["..", "#."], 2) == 4
    assert s.numberOfRoutes(["#"], 750) == 0
    assert s.numberOfRoutes([".."], 1) == 4
    print("LC 3797 OK")


if __name__ == "__main__":
    _test()
