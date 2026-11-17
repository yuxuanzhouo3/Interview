"""
LC 2827. Number of Beautiful Integers in the Range — Hard
https://leetcode.com/problems/number-of-beautiful-integers-in-the-range/
Best: Digit DP odd/even digit balance — O(L*diff*2)
"""

class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def calc(n: int) -> int:
            if n < 0:
                return 0
            s = str(n)
            m = len(s)
            from functools import lru_cache

            @lru_cache(None)
            def dfs(pos, diff, mod, tight, started):
                # diff = even_count - odd_count + offset; use even-odd
                if pos == m:
                    return int(started and diff == 0 and mod == 0)
                up = int(s[pos]) if tight else 9
                ans = 0
                for d in range(up + 1):
                    nt = tight and d == up
                    if not started and d == 0:
                        ans += dfs(pos + 1, diff, mod, nt, False)
                    else:
                        nd = diff + (1 if d % 2 == 0 else -1)
                        nm = (mod * 10 + d) % k
                        ans += dfs(pos + 1, nd, nm, nt, True)
                return ans

            return dfs(0, 0, 0, True, False)

        return calc(high) - calc(low - 1)


def _test():
    s = Solution()
    assert s.numberOfBeautifulIntegers(10, 20, 3) == 2
    assert s.numberOfBeautifulIntegers(1, 10, 1) == 1
    assert s.numberOfBeautifulIntegers(5, 5, 2) == 0
    print("LC 2827 OK")


if __name__ == "__main__":
    _test()

