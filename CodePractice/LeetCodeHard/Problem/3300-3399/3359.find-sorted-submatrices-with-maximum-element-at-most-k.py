"""
LC 3359. Find Sorted Submatrices With Maximum Element at Most K — Hard
https://leetcode.com/problems/find-sorted-submatrices-with-maximum-element-at-most-k/
Best: Per-row histogram of valid heights + monotonic stack — O(mn)
"""

from typing import List


class Solution:
    def countSubmatrices(self, grid: List[List[int]], k: int) -> int:
        # Count submatrices that are row-wise sorted non-decreasing and max <= k
        m, n = len(grid), len(grid[0])
        # For each cell, consecutive valid sorted length leftward including itself
        left = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if grid[i][j] > k:
                    left[i][j] = 0
                elif j > 0 and grid[i][j] >= grid[i][j - 1] and left[i][j - 1] > 0:
                    left[i][j] = left[i][j - 1] + 1
                else:
                    left[i][j] = 1 if grid[i][j] <= k else 0

        ans = 0
        for j in range(n):
            # histogram of left[:, j]; count submatrices ending at column j
            stack = []  # (height, width_contrib)
            total = 0
            for i in range(m):
                h = left[i][j]
                cnt = 1
                while stack and stack[-1][0] >= h:
                    ph, pc = stack.pop()
                    total -= ph * pc
                    cnt += pc
                stack.append((h, cnt))
                total += h * cnt
                ans += total
        return ans


def _test():
    s = Solution()
    # All 1s 2x2, k=1 → all submatrices sorted, max<=1: 4+2+2+1=9? 
    # 1x1:4, 1x2:2, 2x1:2, 2x2:1 = 9
    assert s.countSubmatrices([[1, 1], [1, 1]], 1) == 9
    assert s.countSubmatrices([[2, 1]], 1) == 1  # only [1]
    print("LC 3359 OK")


if __name__ == "__main__":
    _test()
