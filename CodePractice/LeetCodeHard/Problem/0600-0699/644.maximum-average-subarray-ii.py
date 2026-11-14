"""
LC 644. Maximum Average Subarray II — Hard
https://leetcode.com/problems/maximum-average-subarray-ii/
Best: binary search on average + prefix min — O(n log W)
"""

from typing import List


class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        def check(mid: float) -> bool:
            # exists subarray len>=k with avg >= mid iff sum(a_i - mid) >= 0
            pref = 0.0
            prev = 0.0
            min_pref = 0.0
            for i in range(k):
                pref += nums[i] - mid
            if pref >= 0:
                return True
            for i in range(k, len(nums)):
                pref += nums[i] - mid
                prev += nums[i - k] - mid
                min_pref = min(min_pref, prev)
                if pref - min_pref >= 0:
                    return True
            return False

        lo, hi = float(min(nums)), float(max(nums))
        for _ in range(80):
            mid = (lo + hi) / 2
            if check(mid):
                lo = mid
            else:
                hi = mid
        return lo


def _test():
    s = Solution()
    ans = s.findMaxAverage([1, 12, -5, -6, 50, 3], 4)
    assert abs(ans - 12.75) < 1e-4
    ans2 = s.findMaxAverage([5], 1)
    assert abs(ans2 - 5.0) < 1e-5
    print("LC 644 OK")


if __name__ == "__main__":
    _test()
