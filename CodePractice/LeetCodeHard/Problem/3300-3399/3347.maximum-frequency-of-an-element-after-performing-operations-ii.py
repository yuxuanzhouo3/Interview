"""
LC 3347. Maximum Frequency of an Element After Performing Operations II — Hard
https://leetcode.com/problems/maximum-frequency-of-an-element-after-performing-operations-ii/
Best: Sort + sliding window / candidates — O(n log n)
"""

from typing import List
from bisect import bisect_left, bisect_right


class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        nums.sort()
        n = len(nums)
        # candidate targets: each nums[i], nums[i]-k, nums[i]+k
        cands = set()
        for x in nums:
            cands.add(x)
            cands.add(x - k)
            cands.add(x + k)
        ans = 1
        for t in cands:
            # elements already == t
            lo = bisect_left(nums, t)
            hi = bisect_right(nums, t)
            already = hi - lo
            # elements in [t-k, t+k] can become t
            L = bisect_left(nums, t - k)
            R = bisect_right(nums, t + k)
            can = R - L
            ans = max(ans, min(can, already + numOperations))
        return ans


def _test():
    s = Solution()
    assert s.maxFrequency([1, 4, 5], 1, 2) == 2
    assert s.maxFrequency([5, 11, 20, 20], 5, 1) == 2
    print("LC 3347 OK")


if __name__ == "__main__":
    _test()
