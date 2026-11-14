"""
LC 719. Find K-th Smallest Pair Distance — Hard
Best: Sort + binary search count pairs — O(n log n + n log W)
"""

from typing import List


class Solution:
    def smallestDistancePair(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)

        def count(mid):
            cnt = left = 0
            for right in range(n):
                while nums[right] - nums[left] > mid:
                    left += 1
                cnt += right - left
            return cnt

        lo, hi = 0, nums[-1] - nums[0]
        while lo < hi:
            mid = (lo + hi) // 2
            if count(mid) >= k:
                hi = mid
            else:
                lo = mid + 1
        return lo


def _test():
    s = Solution()
    assert s.smallestDistancePair([1, 3, 1], 1) == 0
    assert s.smallestDistancePair([1, 1, 1], 2) == 0
    print("LC 719 OK")


if __name__ == "__main__":
    _test()
