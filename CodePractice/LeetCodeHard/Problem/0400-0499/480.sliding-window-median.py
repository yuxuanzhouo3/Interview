"""
LC 480. Sliding Window Median — Hard
https://leetcode.com/problems/sliding-window-median/
Best: Dual heap + lazy deletion — O(n log k)
"""

from typing import List
import heapq
from collections import defaultdict


class Solution:
    def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        small = []  # max-heap (negated)
        large = []  # min-heap
        delayed = defaultdict(int)
        n_small = n_large = 0

        def prune(heap):
            while heap:
                x = -heap[0] if heap is small else heap[0]
                if delayed[x]:
                    delayed[x] -= 1
                    heapq.heappop(heap)
                else:
                    break

        def balance():
            nonlocal n_small, n_large
            if n_small > n_large + 1:
                heapq.heappush(large, -heapq.heappop(small))
                n_small -= 1
                n_large += 1
                prune(small)
            elif n_small < n_large:
                heapq.heappush(small, -heapq.heappop(large))
                n_large -= 1
                n_small += 1
                prune(large)

        def add(num):
            nonlocal n_small, n_large
            if not small or num <= -small[0]:
                heapq.heappush(small, -num)
                n_small += 1
            else:
                heapq.heappush(large, num)
                n_large += 1
            balance()

        def remove(num):
            nonlocal n_small, n_large
            delayed[num] += 1
            if num <= -small[0]:
                n_small -= 1
                if num == -small[0]:
                    prune(small)
            else:
                n_large -= 1
                if large and num == large[0]:
                    prune(large)
            balance()

        def get_median():
            if k & 1:
                return float(-small[0])
            return (-small[0] + large[0]) / 2.0

        ans = []
        for i, num in enumerate(nums):
            add(num)
            if i >= k - 1:
                ans.append(get_median())
                remove(nums[i - k + 1])
        return ans


def _test():
    s = Solution()
    assert s.medianSlidingWindow([1, 3, -1, -3, 5, 3, 6, 7], 3) == [
        1.0,
        -1.0,
        -1.0,
        3.0,
        5.0,
        6.0,
    ]
    assert s.medianSlidingWindow([1, 2, 3, 4], 4) == [2.5]
    assert s.medianSlidingWindow([1], 1) == [1.0]
    print("LC 480 OK")


if __name__ == "__main__":
    _test()
