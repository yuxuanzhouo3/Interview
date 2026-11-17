"""
LC 2147. Number of Ways to Divide a Long Corridor — Hard
https://leetcode.com/problems/number-of-ways-to-divide-a-long-corridor/
Best: seat gaps between pairs — O(n)
"""

from typing import List


class Solution:
    def numberOfWays(self, corridor: str) -> int:
        MOD = 10**9 + 7
        seats = [i for i, c in enumerate(corridor) if c == "S"]
        m = len(seats)
        if m < 2 or m % 2:
            return 0
        ans = 1
        for i in range(2, m, 2):
            ans = ans * (seats[i] - seats[i - 1]) % MOD
        return ans


def _test():
    s = Solution()
    assert s.numberOfWays("SSPPSPS") == 3
    assert s.numberOfWays("PPSPSP") == 1
    assert s.numberOfWays("S") == 0
    print("LC 2147 OK")


if __name__ == "__main__":
    _test()
