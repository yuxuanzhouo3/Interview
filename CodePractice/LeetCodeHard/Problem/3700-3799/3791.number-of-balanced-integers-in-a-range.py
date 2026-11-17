"""
LC 3791. Number of Balanced Integers in a Range — Hard
https://leetcode.com/problems/number-of-balanced-integers-in-a-range/
Best: doocs/leetcode reference solution (verified on examples)
"""

from functools import cache
class Solution:
    def countBalanced(self, low: int, high: int) -> int:
        @cache
        def dfs(pos: int, diff: int, lim: int) -> int:
            if pos >= len(num):
                return 1 if diff == 0 else 0
            res = 0
            up = int(num[pos]) if lim else 9
            for i in range(up + 1):
                res += dfs(
                    pos + 1, diff + i * (1 if pos % 2 == 0 else -1), lim and i == up
                )
            return res

        if high < 11:
            return 0
        low = max(low, 11)
        num = str(low - 1)
        a = dfs(0, 0, True)
        dfs.cache_clear()
        num = str(high)
        b = dfs(0, 0, True)
        return b - a

def _test():
    s = Solution()
    assert s.countBalanced(1, 100) == 9
    assert s.countBalanced(120, 129) == 1
    assert s.countBalanced(1234, 1234) == 0
    print("LC 3791 OK")


if __name__ == "__main__":
    _test()
