"""
LC 1900. The Earliest and Latest Rounds Where Players Compete — Hard
https://leetcode.com/problems/the-earliest-and-latest-rounds-where-players-compete/
Best: 记忆化枚举晋级位置 — O(poly(n)) with n<=28
"""

from typing import List
from functools import lru_cache
from math import inf


class Solution:
    def earliestAndLatest(self, n: int, firstPlayer: int, secondPlayer: int) -> List[int]:
        @lru_cache(None)
        def dfs(l: int, r: int, m: int):
            if l > r:
                l, r = r, l
            if l + r == m - 1:
                return (1, 1)
            res = [inf, -inf]
            half = m >> 1
            for mask in range(1 << half):
                win = [False] * m
                for j in range(half):
                    if mask >> j & 1:
                        win[j] = True
                    else:
                        win[m - 1 - j] = True
                if m & 1:
                    win[half] = True
                win[m - 1 - l] = win[m - 1 - r] = False
                win[l] = win[r] = True
                a = b = c = 0
                for j in range(m):
                    if j == l:
                        a = c
                    if j == r:
                        b = c
                    if win[j]:
                        c += 1
                x, y = dfs(a, b, c)
                res[0] = min(res[0], x + 1)
                res[1] = max(res[1], y + 1)
            return (res[0], res[1])

        return list(dfs(firstPlayer - 1, secondPlayer - 1, n))


def _test():
    s = Solution()
    assert s.earliestAndLatest(11, 2, 4) == [3, 4]
    assert s.earliestAndLatest(5, 1, 5) == [1, 1]
    print("LC 1900 OK")


if __name__ == "__main__":
    _test()
