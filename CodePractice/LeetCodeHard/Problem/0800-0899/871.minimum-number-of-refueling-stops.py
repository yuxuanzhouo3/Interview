"""
LC 871. Minimum Number of Refueling Stops — Hard
Best: Max-heap of passed fuels — O(n log n)
"""

from typing import List
import heapq


class Solution:
    def minRefuelStops(self, target: int, startFuel: int, stations: List[List[int]]) -> int:
        heap = []
        fuel = startFuel
        ans = prev = 0
        stations = stations + [[target, 0]]
        for pos, f in stations:
            fuel -= pos - prev
            while fuel < 0 and heap:
                fuel += -heapq.heappop(heap)
                ans += 1
            if fuel < 0:
                return -1
            heapq.heappush(heap, -f)
            prev = pos
        return ans


def _test():
    s = Solution()
    assert s.minRefuelStops(1, 1, []) == 0
    assert s.minRefuelStops(100, 1, [[10, 100]]) == -1
    assert s.minRefuelStops(100, 10, [[10, 60], [20, 30], [30, 30], [60, 40]]) == 2
    print("LC 871 OK")


if __name__ == "__main__":
    _test()
