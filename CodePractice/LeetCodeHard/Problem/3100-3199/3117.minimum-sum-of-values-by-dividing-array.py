"""
LC 3117. Minimum Sum of Values by Dividing Array — Hard
https://leetcode.com/problems/minimum-sum-of-values-by-dividing-array/
Best: Memo DFS on (i,j,AND) — O(n m log M)
"""

from functools import cache
from math import inf
from typing import List


class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n, m = len(nums), len(andValues)

        @cache
        def dfs(i: int, j: int, a: int) -> int:
            if n - i < m - j:
                return inf
            if j == m:
                return 0 if i == n else inf
            a &= nums[i]
            if a < andValues[j]:
                return inf
            ans = dfs(i + 1, j, a)
            if a == andValues[j]:
                ans = min(ans, dfs(i + 1, j + 1, -1) + nums[i])
            return ans

        ans = dfs(0, 0, -1)
        return -1 if ans == inf else ans


def _test():
    s = Solution()
    assert s.minimumValueSum([1, 4, 3, 3, 2], [0, 3, 3, 2]) == 12
    assert s.minimumValueSum([2, 3, 5, 7, 7, 7, 5], [0, 7, 5]) == 17
    assert s.minimumValueSum([1, 2, 3, 4], [2]) == -1
    print("LC 3117 OK")


if __name__ == "__main__":
    _test()
