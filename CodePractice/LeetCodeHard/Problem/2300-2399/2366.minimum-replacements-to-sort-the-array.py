"""
LC 2366. Minimum Replacements to Sort the Array — Hard
https://leetcode.com/problems/minimum-replacements-to-sort-the-array/
Best: Right-to-left greedy split — O(n)
"""

from typing import List


class Solution:
    def minimumReplacement(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        prev = nums[-1]
        for i in range(n - 2, -1, -1):
            if nums[i] <= prev:
                prev = nums[i]
                continue
            k = (nums[i] + prev - 1) // prev
            ans += k - 1
            prev = nums[i] // k
        return ans


def _test():
    s = Solution()
    assert s.minimumReplacement([3, 9, 3]) == 2
    assert s.minimumReplacement([1, 2, 3, 4, 5]) == 0
    assert s.minimumReplacement([19, 7]) == 2
    print("LC 2366 OK")


if __name__ == "__main__":
    _test()
