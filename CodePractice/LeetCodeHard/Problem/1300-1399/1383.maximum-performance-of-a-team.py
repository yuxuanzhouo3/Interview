"""
LC 1383. Maximum Performance of a Team — Hard
https://leetcode.com/problems/maximum-performance-of-a-team/
Best: Sort by efficiency + min-heap on speed — O(n log n)
"""

import heapq
from typing import List


class Solution:
    def maxPerformance(self, n: int, speed: List[int], efficiency: List[int], k: int) -> int:
        MOD = 10**9 + 7
        people = sorted(zip(efficiency, speed), reverse=True)
        heap = []
        spd_sum = 0
        ans = 0
        for eff, spd in people:
            heapq.heappush(heap, spd)
            spd_sum += spd
            if len(heap) > k:
                spd_sum -= heapq.heappop(heap)
            ans = max(ans, spd_sum * eff)
        return ans % MOD


def _test():
    s = Solution()
    assert s.maxPerformance(6, [2, 10, 3, 1, 5, 8], [5, 4, 3, 9, 7, 2], 2) == 60
    assert s.maxPerformance(6, [2, 10, 3, 1, 5, 8], [5, 4, 3, 9, 7, 2], 3) == 68
    assert s.maxPerformance(6, [2, 10, 3, 1, 5, 8], [5, 4, 3, 9, 7, 2], 4) == 72
    print("LC 1383 OK")


if __name__ == "__main__":
    _test()
