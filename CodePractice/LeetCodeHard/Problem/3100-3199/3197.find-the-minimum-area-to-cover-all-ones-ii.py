"""
LC 3197. Find the Minimum Area to Cover All Ones II — Hard
https://leetcode.com/problems/find-the-minimum-area-to-cover-all-ones-ii/
Best: Enumerate split lines + bbox — O(m^2 n^2)
"""

from typing import List


class Solution:
    def minimumSum(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])

        def area(r1, r2, c1, c2):
            minr, maxr, minc, maxc = m, -1, n, -1
            for i in range(r1, r2 + 1):
                for j in range(c1, c2 + 1):
                    if grid[i][j] == 1:
                        minr, maxr = min(minr, i), max(maxr, i)
                        minc, maxc = min(minc, j), max(maxc, j)
            if maxr < 0:
                return 0  # empty — invalid for our use if ones exist elsewhere
            return (maxr - minr + 1) * (maxc - minc + 1)

        def bbox_area(cells):
            if not cells:
                return None
            minr = min(i for i, j in cells)
            maxr = max(i for i, j in cells)
            minc = min(j for i, j in cells)
            maxc = max(j for i, j in cells)
            return (maxr - minr + 1) * (maxc - minc + 1)

        ones = [(i, j) for i in range(m) for j in range(n) if grid[i][j] == 1]
        ans = m * n * 3

        # rotate helper: try all split into 3 rectangles by 2 guillotine cuts
        def solve(g):
            R, C = len(g), len(g[0])
            res = R * C * 3

            def ar(r1, r2, c1, c2):
                minr, maxr, minc, maxc = R, -1, C, -1
                for i in range(r1, r2 + 1):
                    for j in range(c1, c2 + 1):
                        if g[i][j]:
                            minr, maxr = min(minr, i), max(maxr, i)
                            minc, maxc = min(minc, j), max(maxc, j)
                if maxr < 0:
                    return None
                return (maxr - minr + 1) * (maxc - minc + 1)

            # two horizontal cuts -> 3 horizontal strips
            for i in range(R - 2):
                for j in range(i + 1, R - 1):
                    a, b, c = ar(0, i, 0, C - 1), ar(i + 1, j, 0, C - 1), ar(j + 1, R - 1, 0, C - 1)
                    if None not in (a, b, c):
                        res = min(res, a + b + c)
            # two vertical cuts
            for i in range(C - 2):
                for j in range(i + 1, C - 1):
                    a, b, c = ar(0, R - 1, 0, i), ar(0, R - 1, i + 1, j), ar(0, R - 1, j + 1, C - 1)
                    if None not in (a, b, c):
                        res = min(res, a + b + c)
            # one horizontal then vertical on one side
            for i in range(R - 1):
                top = ar(0, i, 0, C - 1)
                if top is None:
                    continue
                for j in range(C - 1):
                    a, b = ar(i + 1, R - 1, 0, j), ar(i + 1, R - 1, j + 1, C - 1)
                    if None not in (a, b):
                        res = min(res, top + a + b)
                bot = ar(i + 1, R - 1, 0, C - 1)
                if bot is None:
                    continue
                for j in range(C - 1):
                    a, b = ar(0, i, 0, j), ar(0, i, j + 1, C - 1)
                    if None not in (a, b):
                        res = min(res, bot + a + b)
            # one vertical then horizontal on one side
            for i in range(C - 1):
                left = ar(0, R - 1, 0, i)
                if left is None:
                    continue
                for j in range(R - 1):
                    a, b = ar(0, j, i + 1, C - 1), ar(j + 1, R - 1, i + 1, C - 1)
                    if None not in (a, b):
                        res = min(res, left + a + b)
                right = ar(0, R - 1, i + 1, C - 1)
                if right is None:
                    continue
                for j in range(R - 1):
                    a, b = ar(0, j, 0, i), ar(j + 1, R - 1, 0, i)
                    if None not in (a, b):
                        res = min(res, right + a + b)
            return res

        ans = solve(grid)
        # rotate
        rot = [[grid[i][j] for i in range(m)] for j in range(n - 1, -1, -1)]
        # actually 90° rotate
        rot = [[grid[m - 1 - j][i] for j in range(m)] for i in range(n)]
        # wait proper: rot[r][c] = grid[m-1-c][r] with size n x m
        rot = [[grid[m - 1 - c][r] for c in range(m)] for r in range(n)]
        ans = min(ans, solve(rot))
        return ans


def _test():
    s = Solution()
    assert s.minimumSum([[1, 0, 1], [1, 1, 1]]) == 5
    assert s.minimumSum([[1, 0, 1, 0], [0, 1, 0, 1]]) == 5
    print("LC 3197 OK")


if __name__ == "__main__":
    _test()
