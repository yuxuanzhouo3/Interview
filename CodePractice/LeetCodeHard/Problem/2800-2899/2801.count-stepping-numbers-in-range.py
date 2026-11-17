"""
LC 2801. Count Stepping Numbers in Range — Hard
https://leetcode.com/problems/count-stepping-numbers-in-range/
Best: Digit DP stepping numbers — O(L*10)
"""

class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7

        def calc(s: str) -> int:
            n = len(s)
            from functools import lru_cache

            @lru_cache(None)
            def dfs(pos, prev, tight, started):
                if pos == n:
                    return int(started)
                up = int(s[pos]) if tight else 9
                ans = 0
                for d in range(up + 1):
                    nt = tight and d == up
                    if not started:
                        if d == 0:
                            ans += dfs(pos + 1, prev, nt, False)
                        else:
                            ans += dfs(pos + 1, d, nt, True)
                    else:
                        if abs(d - prev) == 1:
                            ans += dfs(pos + 1, d, nt, True)
                return ans % MOD

            return dfs(0, 0, True, False)

        def dec(s: str) -> str:
            a = list(s)
            i = len(a) - 1
            while i >= 0 and a[i] == '0':
                a[i] = '9'
                i -= 1
            a[i] = str(int(a[i]) - 1)
            t = ''.join(a).lstrip('0')
            return t or '0'

        return (calc(high) - calc(dec(low))) % MOD


def _test():
    s = Solution()
    assert s.countSteppingNumbers("1", "11") == 10
    assert s.countSteppingNumbers("90", "101") == 2
    print("LC 2801 OK")


if __name__ == "__main__":
    _test()

