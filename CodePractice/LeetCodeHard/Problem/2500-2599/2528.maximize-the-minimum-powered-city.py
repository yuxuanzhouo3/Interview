"""
LC 2528. Maximize the Minimum Powered City — Hard
https://leetcode.com/problems/maximize-the-minimum-powered-city/
Best: Binary search + diff greedy — O(n log S)
"""

from typing import List


class Solution:
    def maxPower(self, stations: List[int], r: int, k: int) -> int:
        n = len(stations)
        diff = [0] * (n + 1)
        for i, v in enumerate(stations):
            L, R = max(0, i - r), min(n - 1, i + r)
            diff[L] += v
            diff[R + 1] -= v
        power = [0] * n
        cur = 0
        for i in range(n):
            cur += diff[i]
            power[i] = cur

        def ok(mid: int) -> bool:
            need = 0
            add = [0] * (n + 1)
            run = 0
            for i in range(n):
                run += add[i]
                have = power[i] + run
                if have < mid:
                    d = mid - have
                    need += d
                    if need > k:
                        return False
                    run += d
                    end = min(n - 1, i + 2 * r)
                    add[end + 1] -= d
            return True

        lo, hi = 0, max(power) + k
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if ok(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo


def _test():
    s = Solution()
    assert s.maxPower([1, 2, 4, 5, 0], 1, 2) == 5
    assert s.maxPower([4, 4, 4, 4], 0, 3) == 4
    print("LC 2528 OK")


if __name__ == "__main__":
    _test()

