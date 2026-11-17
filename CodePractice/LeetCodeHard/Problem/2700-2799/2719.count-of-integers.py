"""
LC 2719. Count of Integers — Hard
https://leetcode.com/problems/count-of-integers/
Best: Digit DP on digit-sum range — O(len*sum*2)
"""

class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7

        def calc(s: str) -> int:
            n = len(s)
            from functools import lru_cache

            @lru_cache(None)
            def dfs(pos, sm, tight, started):
                if sm > max_sum:
                    return 0
                if pos == n:
                    return int(min_sum <= sm <= max_sum)
                up = int(s[pos]) if tight else 9
                ans = 0
                for d in range(up + 1):
                    ans += dfs(pos + 1, sm + d, tight and d == up, started or d > 0)
                return ans % MOD

            return dfs(0, 0, True, False)

        def dec(s: str) -> str:
            a = list(s)
            i = len(a) - 1
            while i >= 0 and a[i] == '0':
                a[i] = '9'
                i -= 1
            a[i] = str(int(a[i]) - 1)
            return ''.join(a).lstrip('0') or '0'

        return (calc(num2) - calc(dec(num1))) % MOD


def _test():
    s = Solution()
    assert s.count("1", "12", 1, 8) == 11
    assert s.count("1", "5", 1, 5) == 5
    print("LC 2719 OK")


if __name__ == "__main__":
    _test()

