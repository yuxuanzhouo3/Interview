"""
LC 1210. Minimum Moves to Reach Target with Rotations — Hard
https://leetcode.com/problems/minimum-moves-to-reach-target-with-rotations/
Best: BFS on (tail_r, tail_c, orient) — O(n^2)
"""

from collections import deque
from typing import List


class Solution:
    def minimumMoves(self, grid: List[List[int]]) -> int:
        n = len(grid)
        # state: (tr, tc, dr)  dr=0 horizontal, 1 vertical; head = tail + (0,1) or (1,0)
        start = (0, 0, 0)
        target = (n - 1, n - 2, 0)
        q = deque([(0, 0, 0, 0)])  # tr,tc,dr,steps
        seen = {(0, 0, 0)}

        def empty(r, c):
            return 0 <= r < n and 0 <= c < n and grid[r][c] == 0

        while q:
            tr, tc, dr, steps = q.popleft()
            if (tr, tc, dr) == target:
                return steps
            # crawl forward
            if dr == 0:  # horizontal
                # right
                if empty(tr, tc + 2):
                    nxt = (tr, tc + 1, 0)
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append((*nxt, steps + 1))
                # down
                if empty(tr + 1, tc) and empty(tr + 1, tc + 1):
                    nxt = (tr + 1, tc, 0)
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append((*nxt, steps + 1))
                # clockwise
                if empty(tr + 1, tc) and empty(tr + 1, tc + 1):
                    nxt = (tr, tc, 1)
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append((*nxt, steps + 1))
            else:  # vertical
                # down
                if empty(tr + 2, tc):
                    nxt = (tr + 1, tc, 1)
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append((*nxt, steps + 1))
                # right
                if empty(tr, tc + 1) and empty(tr + 1, tc + 1):
                    nxt = (tr, tc + 1, 1)
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append((*nxt, steps + 1))
                # counterclockwise
                if empty(tr, tc + 1) and empty(tr + 1, tc + 1):
                    nxt = (tr, tc, 0)
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append((*nxt, steps + 1))
        return -1


def _test():
    s = Solution()
    grid = [
        [0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 1, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
    ]
    assert s.minimumMoves(grid) == 11
    grid2 = [
        [0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 1],
        [1, 1, 1, 0, 0, 1],
        [1, 1, 1, 0, 0, 0],
    ]
    assert s.minimumMoves(grid2) == 9
    print("LC 1210 OK")


if __name__ == "__main__":
    _test()
