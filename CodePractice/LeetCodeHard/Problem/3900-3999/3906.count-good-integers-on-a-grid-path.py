"""
LC 3906. Count Good Integers on a Grid Path — Hard
https://leetcode.com/problems/count-good-integers-on-a-grid-path/
Best: doocs/leetcode reference solution (verified on examples)
"""

from functools import cache
class Solution:
    def countGoodIntegersOnPath(self, l: int, r: int, directions: str) -> int:
        key = [False] * 16
        row, col = 0, 0
        key[0] = True
        for c in directions:
            if c == "D":
                row += 1
            else:
                col += 1
            key[row * 4 + col] = True

        s = ""

        @cache
        def dfs(pos, last, lim):
            if pos == 16:
                return 1

            res = 0
            start = last if key[pos] else 0
            end = int(s[pos]) if lim else 9

            for i in range(start, end + 1):
                res += dfs(pos + 1, i if key[pos] else last, lim and (i == end))

            return res

        def calc(x):
            nonlocal s
            if x < 0:
                return 0
            s = str(x).zfill(16)
            dfs.cache_clear()
            return dfs(0, 0, True)

        return calc(r) - calc(l - 1)

def _test():
    s = Solution()
    assert s.countGoodIntegersOnPath(8, 10, "DDDRRR") == 2
    assert s.countGoodIntegersOnPath(123456789, 123456790, "DDRRDR") == 1
    assert s.countGoodIntegersOnPath(1288561398769758, 1288561398769758, "RRRDDD") == 0
    print("LC 3906 OK")


if __name__ == "__main__":
    _test()
