"""
LC 675. Cut Off Trees for Golf Event — Hard
https://leetcode.com/problems/cut-off-trees-for-golf-event/
Best: sort trees + BFS between — O(t * mn)
"""

from collections import deque
from typing import List


class Solution:
    def cutOffTree(self, forest: List[List[int]]) -> int:
        if not forest or not forest[0]:
            return 0
        m, n = len(forest), len(forest[0])
        trees = sorted(
            (forest[i][j], i, j)
            for i in range(m)
            for j in range(n)
            if forest[i][j] > 1
        )

        def bfs(si, sj, ti, tj):
            if (si, sj) == (ti, tj):
                return 0
            q = deque([(si, sj, 0)])
            seen = {(si, sj)}
            while q:
                x, y, d = q.popleft()
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < m and 0 <= ny < n and (nx, ny) not in seen and forest[nx][ny]:
                        if (nx, ny) == (ti, tj):
                            return d + 1
                        seen.add((nx, ny))
                        q.append((nx, ny, d + 1))
            return -1

        ans = 0
        cx = cy = 0
        for _, tx, ty in trees:
            d = bfs(cx, cy, tx, ty)
            if d < 0:
                return -1
            ans += d
            cx, cy = tx, ty
        return ans


def _test():
    s = Solution()
    assert s.cutOffTree([[1, 2, 3], [0, 0, 4], [7, 6, 5]]) == 6
    assert s.cutOffTree([[1, 2, 3], [0, 0, 0], [7, 6, 5]]) == -1
    assert s.cutOffTree([[2, 3, 4], [0, 0, 5], [8, 7, 6]]) == 6
    print("LC 675 OK")


if __name__ == "__main__":
    _test()
