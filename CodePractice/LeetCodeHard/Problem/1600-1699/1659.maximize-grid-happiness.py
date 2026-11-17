"""
LC 1659. Maximize Grid Happiness — Hard
https://leetcode.com/problems/maximize-grid-happiness/
Best: Row-profile DP — O(m · 3^n · (I+E))
"""

from functools import lru_cache


class Solution:
    def getMaxGridHappiness(
        self, m: int, n: int, introvertsCount: int, extrovertsCount: int
    ) -> int:
        P = 3**n
        # decode profile to list of 0/1/2
        def decode(mask):
            cells = []
            for _ in range(n):
                cells.append(mask % 3)
                mask //= 3
            return cells

        profiles = [decode(p) for p in range(P)]

        def cost(up, cur):
            # happiness from placing cur row given up row
            h = 0
            for j in range(n):
                if cur[j] == 1:
                    h += 120
                elif cur[j] == 2:
                    h += 40
                if cur[j] and up[j]:
                    # vertical neighbor
                    h += delta(cur[j], up[j])
                    h += delta(up[j], cur[j])
                if j and cur[j] and cur[j - 1]:
                    h += delta(cur[j], cur[j - 1])
                    h += delta(cur[j - 1], cur[j])
            return h

        def delta(a, b):
            # effect on a when neighbor b exists
            if a == 1:
                return -30
            if a == 2:
                return 20
            return 0

        def count_people(cells):
            i = e = 0
            for c in cells:
                if c == 1:
                    i += 1
                elif c == 2:
                    e += 1
            return i, e

        @lru_cache(None)
        def dp(row, up, ii, ee):
            if row == m:
                return 0
            best = 0
            for p in range(P):
                cells = profiles[p]
                ni, ne = count_people(cells)
                if ni > ii or ne > ee:
                    continue
                best = max(
                    best,
                    cost(profiles[up] if row else [0] * n, cells)
                    + dp(row + 1, p, ii - ni, ee - ne),
                )
            return best

        return dp(0, 0, introvertsCount, extrovertsCount)


def _test():
    s = Solution()
    assert s.getMaxGridHappiness(2, 3, 1, 2) == 240
    assert s.getMaxGridHappiness(3, 1, 2, 1) == 260
    assert s.getMaxGridHappiness(2, 2, 4, 0) == 240
    print("LC 1659 OK")


if __name__ == "__main__":
    _test()
