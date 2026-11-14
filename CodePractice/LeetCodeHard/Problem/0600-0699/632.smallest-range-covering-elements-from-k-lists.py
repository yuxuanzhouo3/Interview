"""
LC 632. Smallest Range Covering Elements from K Lists — Hard
https://leetcode.com/problems/smallest-range-covering-elements-from-k-lists/
Best: min-heap k-pointer — O(N log k)
"""

import heapq
from typing import List


class Solution:
    def smallestRange(self, nums: List[List[int]]) -> List[int]:
        heap = []
        cur_max = float("-inf")
        for i, row in enumerate(nums):
            heapq.heappush(heap, (row[0], i, 0))
            cur_max = max(cur_max, row[0])

        best = [float("-inf"), float("inf")]
        while True:
            cur_min, i, j = heapq.heappop(heap)
            if cur_max - cur_min < best[1] - best[0] or (
                cur_max - cur_min == best[1] - best[0] and cur_min < best[0]
            ):
                best = [cur_min, cur_max]
            if j + 1 == len(nums[i]):
                break
            nxt = nums[i][j + 1]
            heapq.heappush(heap, (nxt, i, j + 1))
            cur_max = max(cur_max, nxt)
        return best


def _test():
    s = Solution()
    assert s.smallestRange([[4, 10, 15, 24, 26], [0, 9, 12, 20], [5, 18, 22, 30]]) == [20, 24]
    assert s.smallestRange([[1, 2, 3], [1, 2, 3], [1, 2, 3]]) == [1, 1]
    print("LC 632 OK")


if __name__ == "__main__":
    _test()
