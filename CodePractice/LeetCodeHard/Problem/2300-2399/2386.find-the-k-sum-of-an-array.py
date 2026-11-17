"""
LC 2386. Find the K-Sum of an Array — Hard
https://leetcode.com/problems/find-the-k-sum-of-an-array/
Best: Max subsum + heap next k-sums — O(k log k)
"""


from typing import List
import heapq


class Solution:
    def kSum(self, nums: List[int], k: int) -> int:
        total = sum(x for x in nums if x > 0)
        abs_nums = sorted(abs(x) for x in nums)
        # max subsum is total; generate next smaller by subtracting positives / equivalent
        h = [(-total, 0)]  # (-sum, index into abs_nums to consider)
        for _ in range(k):
            s, i = heapq.heappop(h)
            ans = -s
            if i < len(abs_nums):
                heapq.heappush(h, (s + abs_nums[i], i + 1))
                if i > 0:
                    heapq.heappush(h, (s - abs_nums[i - 1] + abs_nums[i], i + 1))
        return ans


def _test():
    s = Solution()
    assert s.kSum([2, 4, -2], 5) == 2
    assert s.kSum([1, -2, 3, 4, -10, 12], 16) == 10
    print("LC 2386 OK")


if __name__ == "__main__":
    _test()

