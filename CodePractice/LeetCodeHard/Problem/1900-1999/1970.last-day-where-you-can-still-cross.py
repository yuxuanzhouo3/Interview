"""
LC 1970. Last Day Where You Can Still Cross — Hard
https://leetcode.com/problems/last-day-where-you-can-still-cross/
Best: binary search + BFS / Union-Find — O(rc log(rc))
"""

from typing import List
from collections import deque


class Solution:
    def latestDayToCross(self, row: int, col: int, cells: List[List[int]]) -> int:
        def can(day):
            water = set((r - 1, c - 1) for r, c in cells[:day])
            q = deque()
            seen = [[False] * col for _ in range(row)]
            for c in range(col):
                if (0, c) not in water:
                    q.append((0, c))
                    seen[0][c] = True
            while q:
                r, c = q.popleft()
                if r == row - 1:
                    return True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < row and 0 <= nc < col and not seen[nr][nc] and (nr, nc) not in water:
                        seen[nr][nc] = True
                        q.append((nr, nc))
            return False

        lo, hi = 1, len(cells)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if can(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo


def _test():
    s = Solution()
    assert s.latestDayToCross(2, 2, [[1, 1], [2, 1], [1, 2], [2, 2]]) == 2
    assert s.latestDayToCross(2, 2, [[1, 1], [1, 2], [2, 1], [2, 2]]) == 1
    assert s.latestDayToCross(3, 3, [[1, 2], [2, 1], [3, 3], [2, 2], [1, 1], [1, 3], [2, 3], [3, 2], [3, 1]]) == 3
    print("LC 1970 OK")


if __name__ == "__main__":
    _test()
