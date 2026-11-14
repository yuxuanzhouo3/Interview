"""
LC 2302. Count Subarrays With Score Less Than K — Hard
https://leetcode.com/problems/count-subarrays-with-score-less-than-k/
Best: Sliding window — score = sum * length — O(n)
"""

from typing import List


class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        ans = left = cur = 0
        for right, x in enumerate(nums):
            cur += x
            while cur * (right - left + 1) >= k:
                cur -= nums[left]
                left += 1
            ans += right - left + 1
        return ans


def _test():
    s = Solution()
    assert s.countSubarrays([2, 1, 4, 3, 5], 10) == 6
    assert s.countSubarrays([1, 1, 1], 5) == 5
    print("LC 2302 OK")


if __name__ == "__main__":
    _test()
