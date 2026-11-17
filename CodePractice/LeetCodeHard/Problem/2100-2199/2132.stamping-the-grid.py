"""
LC 2132. Stamping the Grid — Hard
https://leetcode.com/problems/stamping-the-grid/
Best: 2D prefix place stamps greedily then cover check — O(mn)
"""


from typing import List


class Solution:
    def possibleToStamp(self, grid: List[List[int]], stampHeight: int, stampWidth: int) -> bool:
        m, n = len(grid), len(grid[0])
        pref = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                pref[i + 1][j + 1] = pref[i + 1][j] + pref[i][j + 1] - pref[i][j] + grid[i][j]

        def rect(x1, y1, x2, y2):
            return pref[x2 + 1][y2 + 1] - pref[x1][y2 + 1] - pref[x2 + 1][y1] + pref[x1][y1]

        diff = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m - stampHeight + 1):
            for j in range(n - stampWidth + 1):
                if rect(i, j, i + stampHeight - 1, j + stampWidth - 1) == 0:
                    diff[i][j] += 1
                    diff[i + stampHeight][j] -= 1
                    diff[i][j + stampWidth] -= 1
                    diff[i + stampHeight][j + stampWidth] += 1
        cover = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                v = diff[i][j]
                if i:
                    v += cover[i - 1][j]
                if j:
                    v += cover[i][j - 1]
                if i and j:
                    v -= cover[i - 1][j - 1]
                cover[i][j] = v
                if grid[i][j] == 0 and cover[i][j] == 0:
                    return False
        return True


def _test():
    s = Solution()
    assert s.possibleToStamp([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], 4, 3) is True
    assert s.possibleToStamp([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], 2, 2) is False
    print("LC 2132 OK")


if __name__ == "__main__":
    _test()

