"""
LC 774. Minimize Max Distance to Gas Station — Hard
https://leetcode.com/problems/minimize-max-distance-to-gas-station/
Best: Binary search on distance — O(n log(W/ε))
"""

from typing import List


class Solution:
    def minmaxGasDist(self, stations: List[int], k: int) -> float:
        def need(d: float) -> int:
            cnt = 0
            for i in range(1, len(stations)):
                gap = stations[i] - stations[i - 1]
                # stations to insert so each piece ≤ d: ceil(gap/d)-1
                cnt += int(gap / d)
            return cnt

        lo, hi = 0.0, float(stations[-1] - stations[0])
        for _ in range(100):
            mid = (lo + hi) / 2
            if need(mid) <= k:
                hi = mid
            else:
                lo = mid
        return hi


def _test():
    s = Solution()
    assert abs(s.minmaxGasDist([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 9) - 0.5) < 1e-5
    assert abs(s.minmaxGasDist([23, 24, 36, 39, 46, 56, 57, 65, 84, 98], 1) - 14.0) < 1e-5
    print("LC 774 OK")


if __name__ == "__main__":
    _test()
