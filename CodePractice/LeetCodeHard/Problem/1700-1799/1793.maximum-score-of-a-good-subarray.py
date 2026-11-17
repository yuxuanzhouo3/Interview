"""
LC 1793. Maximum Score of a Good Subarray — Hard
https://leetcode.com/problems/maximum-score-of-a-good-subarray/
Best: two pointers expand from k — O(n)
"""

from typing import List


class Solution:
    def maximumScore(self, nums: List[int], k: int) -> int:
        n = len(nums)
        l = r = k
        mn = nums[k]
        ans = mn
        while l > 0 or r < n - 1:
            left_val = nums[l - 1] if l > 0 else -1
            right_val = nums[r + 1] if r < n - 1 else -1
            if left_val >= right_val:
                l -= 1
                mn = min(mn, nums[l])
            else:
                r += 1
                mn = min(mn, nums[r])
            ans = max(ans, mn * (r - l + 1))
        return ans


def _test():
    s = Solution()
    assert s.maximumScore([1, 4, 3, 7, 4, 5], 3) == 15
    assert s.maximumScore([5, 5, 4, 5, 4, 1, 1, 1], 0) == 20
    print("LC 1793 OK")


if __name__ == "__main__":
    _test()
