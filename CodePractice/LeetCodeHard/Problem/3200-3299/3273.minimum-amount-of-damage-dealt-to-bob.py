"""
LC 3273. Minimum Amount of Damage Dealt to Bob — Hard
https://leetcode.com/problems/minimum-amount-of-damage-dealt-to-bob/
Best: Greedy sort by damage/time — O(n log n)
"""

from typing import List
from functools import cmp_to_key


class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)
        time = [(health[i] + power - 1) // power for i in range(n)]
        idx = list(range(n))

        def cmp(i, j):
            a = damage[i] * time[j]
            b = damage[j] * time[i]
            return (a < b) - (a > b)

        idx.sort(key=cmp_to_key(cmp))
        ans = elapsed = 0
        for i in idx:
            elapsed += time[i]
            ans += damage[i] * elapsed
        return ans


def _test():
    s = Solution()
    assert s.minDamage(4, [1, 2, 3, 4], [4, 5, 6, 8]) == 39
    assert s.minDamage(1, [1, 1, 1, 1], [1, 2, 3, 4]) == 20
    assert s.minDamage(8, [40], [59]) == 320
    print("LC 3273 OK")


if __name__ == "__main__":
    _test()
