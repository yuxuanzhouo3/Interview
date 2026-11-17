"""
LC 3134. Find the Median of the Uniqueness Array — Hard
https://leetcode.com/problems/find-the-median-of-the-uniqueness-array/
Best: Binary search + sliding window on distinct count — O(n log n)
"""

from typing import List
from collections import defaultdict


class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        total = n * (n + 1) // 2
        need = (total + 1) // 2  # 1-indexed median position

        def at_most(k: int) -> int:
            cnt = defaultdict(int)
            left = 0
            distinct = 0
            ans = 0
            for right, x in enumerate(nums):
                if cnt[x] == 0:
                    distinct += 1
                cnt[x] += 1
                while distinct > k:
                    y = nums[left]
                    cnt[y] -= 1
                    if cnt[y] == 0:
                        distinct -= 1
                    left += 1
                ans += right - left + 1
            return ans

        lo, hi = 1, len(set(nums))
        while lo < hi:
            mid = (lo + hi) // 2
            if at_most(mid) >= need:
                hi = mid
            else:
                lo = mid + 1
        return lo


def _test():
    s = Solution()
    assert s.medianOfUniquenessArray([1, 2, 3]) == 1
    assert s.medianOfUniquenessArray([3, 4, 3, 4, 5]) == 2
    assert s.medianOfUniquenessArray([4, 3, 5, 4]) == 2
    print("LC 3134 OK")


if __name__ == "__main__":
    _test()
