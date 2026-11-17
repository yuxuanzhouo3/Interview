"""
LC 2702. Minimum Operations to Make Numbers Non-positive — Hard
Best: Binary search + greedy — O(n log M)
"""

from typing import List


class Solution:
    def minOperations(self, nums: List[int], x: int, y: int) -> int:
        # each op: all -= y, one extra -= (x-y)
        d = x - y

        def ok(k):
            need = 0
            for v in nums:
                rem = v - k * y
                if rem > 0:
                    need += (rem + d - 1) // d
            return need <= k

        lo, hi = 0, max(nums)
        while lo < hi:
            mid = (lo + hi) // 2
            if ok(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo


def _test():
    s = Solution()
    assert s.minOperations([3, 4, 1, 7, 6], 4, 2) == 3
    print("LC 2702 OK")


if __name__ == "__main__":
    _test()
