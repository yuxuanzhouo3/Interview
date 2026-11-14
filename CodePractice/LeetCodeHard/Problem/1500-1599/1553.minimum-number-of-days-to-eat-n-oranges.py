"""
LC 1553. Minimum Number of Days to Eat N Oranges — Hard
https://leetcode.com/problems/minimum-number-of-days-to-eat-n-oranges/
Best: Memo BFS/DP with /2 /3 jumps — O(log n)
"""

from functools import cache


class Solution:
    def minDays(self, n: int) -> int:
        @cache
        def dp(x):
            if x <= 1:
                return x
            return 1 + min(x % 2 + dp(x // 2), x % 3 + dp(x // 3))

        return dp(n)


def _test():
    s = Solution()
    assert s.minDays(10) == 4
    assert s.minDays(6) == 3
    assert s.minDays(1) == 1
    print("LC 1553 OK")


if __name__ == "__main__":
    _test()
