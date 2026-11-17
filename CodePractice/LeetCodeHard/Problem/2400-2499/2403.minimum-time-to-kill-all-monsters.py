"""
LC 2403. Minimum Time to Kill All Monsters — Hard
https://leetcode.com/problems/minimum-time-to-kill-all-monsters/
Best: Bitmask DP — O(n^2 2^n)
"""
from typing import List
from math import inf


class Solution:
    def minimumTime(self, power: List[int]) -> int:
        n = len(power)
        f = [inf] * (1 << n)
        f[0] = 0
        for mask in range(1 << n):
            gain = bin(mask).count('1') + 1  # current day gain = killed+1
            for i in range(n):
                if not (mask >> i & 1):
                    days = (power[i] + gain - 1) // gain
                    f[mask | (1 << i)] = min(f[mask | (1 << i)], f[mask] + days)
        return f[(1 << n) - 1]


def _test():
    s = Solution()
    assert s.minimumTime([3, 1, 4]) == 4
    assert s.minimumTime([1, 1, 4]) == 4
    assert s.minimumTime([1, 2, 4, 9]) == 6
    print("LC 2403 OK")


if __name__ == "__main__":
    _test()
