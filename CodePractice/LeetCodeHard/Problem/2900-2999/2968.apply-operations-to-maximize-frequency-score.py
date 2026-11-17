"""
LC 2968. Apply Operations to Maximize Frequency Score — Hard
https://leetcode.com/problems/apply-operations-to-maximize-frequency-score/
Best: Sort + binary search length + prefix median cost — O(n log n)
"""

from typing import List


class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        pref = [0] * (n + 1)
        for i, x in enumerate(nums):
            pref[i + 1] = pref[i] + x

        def cost(L, R):
            # cost to make nums[L..R] all equal to median
            m = (L + R) // 2
            left = nums[m] * (m - L) - (pref[m] - pref[L])
            right = (pref[R + 1] - pref[m + 1]) - nums[m] * (R - m)
            return left + right

        ans = 1
        lo, hi = 1, n
        while lo <= hi:
            mid = (lo + hi) // 2
            ok = False
            for i in range(n - mid + 1):
                if cost(i, i + mid - 1) <= k:
                    ok = True
                    break
            if ok:
                ans = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return ans


def _test():
    s = Solution()
    assert s.maxFrequencyScore([1, 2, 6, 4], 3) == 3
    assert s.maxFrequencyScore([1, 4, 4, 2, 4], 0) == 3
    print("LC 2968 OK")


if __name__ == "__main__":
    _test()
