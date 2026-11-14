"""
LC 857. Minimum Cost to Hire K Workers — Hard
Best: Sort by wage/quality ratio + maxheap quality — O(n log n)
"""

from typing import List
import heapq


class Solution:
    def mincostToHireWorkers(self, quality: List[int], wage: List[int], k: int) -> float:
        workers = sorted((w / q, q) for q, w in zip(quality, wage))
        heap = []
        qsum = 0
        ans = float("inf")
        for ratio, q in workers:
            heapq.heappush(heap, -q)
            qsum += q
            if len(heap) > k:
                qsum += heapq.heappop(heap)
            if len(heap) == k:
                ans = min(ans, qsum * ratio)
        return ans


def _test():
    s = Solution()
    assert abs(s.mincostToHireWorkers([10, 20, 5], [70, 50, 30], 2) - 105.0) < 1e-5
    print("LC 857 OK")


if __name__ == "__main__":
    _test()
