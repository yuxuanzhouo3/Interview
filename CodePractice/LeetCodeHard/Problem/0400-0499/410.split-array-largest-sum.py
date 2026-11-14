"""
LC 410. Split Array Largest Sum — Hard
Best: Binary search answer + greedy check — O(n log S)
"""

from typing import List


class Solution:
    def splitArray(self, nums: List[int], k: int) -> int:
        def ok(mid):
            parts = 1
            cur = 0
            for x in nums:
                if cur + x > mid:
                    parts += 1
                    cur = 0
                cur += x
            return parts <= k

        lo, hi = max(nums), sum(nums)
        while lo < hi:
            mid = (lo + hi) // 2
            if ok(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo


def _test():
    s = Solution()
    assert s.splitArray([7, 2, 5, 10, 8], 2) == 18
    assert s.splitArray([1, 2, 3, 4, 5], 2) == 9
    print("LC 410 OK")


if __name__ == "__main__":
    _test()
