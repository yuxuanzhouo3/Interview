"""
LC 1411. Number of Ways to Paint N × 3 Grid — Hard
https://leetcode.com/problems/number-of-ways-to-paint-n-3-grid/
Best: Color types 121 / 123 recurrence — O(n)
"""


class Solution:
    def numOfWays(self, n: int) -> int:
        MOD = 10**9 + 7
        # a: ABA-type (2 colors), b: ABC-type (3 colors)
        a, b = 6, 6
        for _ in range(n - 1):
            a, b = (3 * a + 2 * b) % MOD, (2 * a + 2 * b) % MOD
        return (a + b) % MOD


def _test():
    s = Solution()
    assert s.numOfWays(1) == 12
    assert s.numOfWays(2) == 54
    assert s.numOfWays(7) == 106494
    assert s.numOfWays(5000) == 30228214
    print("LC 1411 OK")


if __name__ == "__main__":
    _test()
