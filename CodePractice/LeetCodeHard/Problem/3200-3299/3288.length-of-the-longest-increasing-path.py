"""
LC 3288. Length of the Longest Increasing Path — Hard
https://leetcode.com/problems/length-of-the-longest-increasing-path/
Best: Sort + Fenwick / LIS on y — O(n log n)
"""

from typing import List
import bisect


class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        # Path through coordinates[k] with strictly increasing x and y.
        kx, ky = coordinates[k]
        left = sorted((x, y) for x, y in coordinates if x < kx and y < ky)
        right = sorted((x, y) for x, y in coordinates if x > kx and y > ky)

        def lis_y(pts: List[tuple]) -> int:
            # pts sorted by x ascending; for same x, process carefully
            # group by x: only one from same x can be used; take them in decreasing y when processing same x
            tails = []
            i = 0
            n = len(pts)
            while i < n:
                j = i
                while j < n and pts[j][0] == pts[i][0]:
                    j += 1
                # same x: consider updates in batch
                updates = []
                for t in range(i, j):
                    y = pts[t][1]
                    pos = bisect.bisect_left(tails, y)
                    updates.append((pos, y))
                for pos, y in updates:
                    if pos == len(tails):
                        # only extend once per x-group with smallest y that extends? 
                        # For LIS with distinct x already, same x cannot both be used.
                        # Apply all replacements but at most one append of the minimum y that extends.
                        pass
                # Correct batch for same x: apply replacements from large y to small, at most one new length
                batch = sorted((pts[t][1] for t in range(i, j)), reverse=True)
                changed = {}
                for y in batch:
                    pos = bisect.bisect_left(tails, y)
                    if pos not in changed:
                        changed[pos] = y
                for pos, y in sorted(changed.items()):
                    if pos == len(tails):
                        tails.append(y)
                    else:
                        tails[pos] = min(tails[pos], y) if False else y
                        # standard: replace tails[pos] with y (y is smaller when we process larger first... )
                        tails[pos] = y
                i = j
            return len(tails)

        return lis_y(left) + 1 + lis_y(right)


def _test():
    s = Solution()
    assert s.maxPathLength([[3, 1], [2, 2], [4, 1], [0, 0], [5, 3]], 1) == 3
    assert s.maxPathLength([[2, 1], [7, 0], [5, 6]], 2) == 2
    print("LC 3288 OK")


if __name__ == "__main__":
    _test()
